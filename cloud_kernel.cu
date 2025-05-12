#include <cuda.h> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#pragma warning(push, 0) 
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>
#pragma warning(pop)

#include <iostream>
#include <vector>
#include <cstring>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct CudaParams {
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraRight;
    glm::vec3 cameraUp;

    glm::vec3 volumeMin;
    glm::vec3 volumeMax;
    glm::vec3 volumeDims;

    int gridDimX;
    int gridDimY;
    int gridDimZ;
    int screenDimX, screenDimY;
    glm::vec3 cellSize;

    
    glm::vec3 sunDirection;
    glm::vec3 sunColor;
    glm::vec3 zenithColor;
    glm::vec3 horizonColor;
    float sunIntensity;

    
    float extinctionScale;
    float anisotropy;

    
    float stepSize;
    float shadowStep;
    float shadowMaxDist;

    float sunExponent;
    float aspectRatio;
    float scale;

    unsigned int noiseSeed; 
    bool isRaining;
    float qcThresholdForRainSpawn;
    float rainExtinctionScale;
    float rainAlbedo;
    float rainFallSpeed;
    float totalSimTime;
    float isoThreshold;     
float surfaceScale;     
float detailNoiseScale;   
float detailWeight;       
float multiScatterScale;  
};


int d_width_render = 0;
int d_height_render = 0;
int d_grid_x_render = 0; 
int d_grid_y_render = 0;
int d_grid_z_render = 0;
float* d_frameBuffer = nullptr;
cudaArray_t d_densityArray = nullptr;
cudaTextureObject_t d_densityTexture = 0;

cudaArray_t d_rainDensityArray = nullptr;
cudaTextureObject_t d_rainDensityTexture = 0;

cudaArray_t d_detailNoiseArray = nullptr;
cudaTextureObject_t d_detailNoiseTexture = 0;


__device__ glm::vec3 mix(const glm::vec3& x, const glm::vec3& y, float a) {
    return x * (1.0f - a) + y * a;
}

__device__ float smoothstep(float edge0, float edge1, float x) {
    x = glm::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// Rain functions:
// Helper for pseudo-randomness
__device__ float hash1(glm::vec2 p) {
    p = glm::fract(p * glm::vec2(123.34f, 345.45f));
    p += glm::dot(p, p + 34.345f);
    return glm::fract(p.x * p.y);
}


__device__ glm::vec2 intersectVolumeDevice(const glm::vec3& rayOrigin, const glm::vec3& rayDirectionINV, const glm::vec3& boxMin, const glm::vec3& boxMax) {
    float tx1 = (boxMin.x - rayOrigin.x) * rayDirectionINV.x;
    float tx2 = (boxMax.x - rayOrigin.x) * rayDirectionINV.x;
    float ty1 = (boxMin.y - rayOrigin.y) * rayDirectionINV.y;
    float ty2 = (boxMax.y - rayOrigin.y) * rayDirectionINV.y;
    float tz1 = (boxMin.z - rayOrigin.z) * rayDirectionINV.z;
    float tz2 = (boxMax.z - rayOrigin.z) * rayDirectionINV.z;
    float tmin = glm::max(glm::min(tx1, tx2), glm::max(glm::min(ty1, ty2), glm::min(tz1, tz2)));
    float tmax = glm::min(glm::max(tx1, tx2), glm::min(glm::max(ty1, ty2), glm::max(tz1, tz2)));

    if (tmin < 0 && tmax > 0) { 
        return glm::vec2(0.0f, tmax); 
    }

    if (tmin >= tmax || tmax < 0) { 
        return glm::vec2(1.0f, 0.0f);
    }

    return glm::vec2{tmin, tmax};
}

// Density sampling using 3D CUDA texture
__device__ float sampleDensity(cudaTextureObject_t densityTex, const glm::vec3& volMin, const glm::vec3& volDims, const glm::vec3& p) {
    // map p to normalized texture coordinates [0,1]
    glm::vec3 texCoords = (p - volMin) / volDims;
    // WIl do trilinear interpolation
    return tex3D<float>(densityTex, texCoords.x, texCoords.y, texCoords.z);
}


__device__ float henyeyGreenstein(float cosTheta, float g) {
    float g2    = g*g;
    float denom = 1.0f + g2 - 2.0f*g*cosTheta;
    return (1.0f - g2) / (4.0f * glm::pi<float>() * powf(denom, 1.5f));
}

__device__ float computeSunTransmittance(cudaTextureObject_t densityTex, const glm::vec3& volMin, const glm::vec3& volDims, const glm::vec3& pos, const glm::vec3& sunDir,
    float shadowStep,
    float shadowMaxDist,
    float extinctionScale)
{
    float Tr = 1.0f;
    float t  = 0.0f;
    while (t < shadowMaxDist && Tr > 0.01f) {
        glm::vec3 p_shadow = pos + sunDir * t;
        float d = sampleDensity(densityTex, volMin, volDims, p_shadow);
        float tau = d * extinctionScale * shadowStep;
        Tr *= __expf(-tau); 
        t   += shadowStep;
    }
    return Tr;
}

__device__ glm::vec3 computeSkyColor(const glm::vec3& dir, const glm::vec3& zenithColor, const glm::vec3& horizonColor, const glm::vec3& sunDir, const glm::vec3& sunColor, float sunExponent, float sunIntensity, bool isRaining)
{   

    if (isRaining == true) {
        glm::vec3 effectiveZenith = zenithColor;
        glm::vec3 effectiveHorizon = horizonColor;
        glm::vec3 effectiveSunColor = sunColor;
        float effectiveSunIntensity = sunIntensity;
        effectiveZenith  = glm::mix(effectiveZenith, glm::vec3(0.4f, 0.4f, 0.45f), 0.6f);
        effectiveHorizon = glm::mix(effectiveHorizon, glm::vec3(0.5f, 0.5f, 0.55f), 0.7f);
        effectiveSunColor *= 0.7f;

        float base = glm::smoothstep(0.0f, 0.4f, dir.y);
        glm::vec3 sky = glm::mix(effectiveZenith, effectiveHorizon, base);

        float sunDot = glm::dot(dir, sunDir);
        float I = powf(glm::max(0.0f, sunDot), sunExponent);
        return sky + effectiveSunColor * I;

    }
    float base = glm::smoothstep(0.0f, 0.4f, dir.y); 
    glm::vec3 sky = glm::mix(zenithColor, horizonColor, base);

    float sunDot = glm::dot(dir, sunDir);
    float I = powf(glm::max(0.0f, sunDot), sunExponent); 
    return sky + sunColor * I;
}
__device__ float fract(float x) {
    return x - floorf(x);
}
__device__ double fade_d(double t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }
__device__ double lerp_d(double t, double a, double b) { return a + t*(b-a); }
__device__ double grad_d(int hash, double x, double y, double z) {
    int h = hash & 15;
    double u = h<8 ? x : y;
    double v = h<4 ? y : (h==12||h==14 ? x : z);
    return (((h&1)==0)?u:-u) + (((h&2)==0)?v:-v);
}




__device__ float computeSunTransmittanceDetail(
    glm::vec3 pos,
cudaTextureObject_t cloudDensityTexture,
cudaTextureObject_t detailNoiseTexture,
    const glm::vec3& sunDir,
    float shadowStep,
    float shadowMaxDist,
    const CudaParams& p)
{
    float t = 0.0f, tr = 1.0f;
    while (t < shadowMaxDist && tr > 0.01f) {
        glm::vec3  sPos = pos + sunDir * t;
        // sample BOTH base + detail
        float bd = sampleDensity(cloudDensityTexture, p.volumeMin, p.volumeDims, sPos);
        float dd = sampleDensity(detailNoiseTexture, p.volumeMin, p.volumeDims, sPos);
        float d = max(bd + dd*p.detailWeight, 0.0f);

        float ext = d * p.extinctionScale;
        tr *= __expf(-ext * shadowStep);
        t  += shadowStep;
    }
    return tr;
}

__global__ void rayTraceKernel(float* frameBuffer, cudaTextureObject_t cloudDensityTexture, cudaTextureObject_t rainDensityTexture, cudaTextureObject_t detailNoiseTexture,CudaParams params)

{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.screenDimX || y >= params.screenDimY) return;

    int idx = (y * params.screenDimX + x) * 3;

    float ndc_x = (2.0f * (x + 0.5f) / params.screenDimX  - 1.0f) * params.aspectRatio * params.scale;
    float ndc_y = (2.0f * (y + 0.5f) / params.screenDimY  - 1.0f) * params.scale;
    glm::vec3 rayDir = glm::normalize(params.cameraFront + params.cameraRight * ndc_x + params.cameraUp * ndc_y);
    glm::vec3 rayOrig = params.cameraPos;

    glm::vec2 inter = intersectVolumeDevice(rayOrig, 1.0f / rayDir, params.volumeMin, params.volumeMin + params.volumeDims);

    if (!(inter.y >= 0 && inter.x < inter.y)) {
        glm::vec3 col = computeSkyColor(rayDir, params.zenithColor, params.horizonColor, params.sunDirection, params.sunColor, params.sunExponent, params.sunIntensity, params.isRaining);
        col = col / (glm::vec3(1.0f)+ col); 
        col = glm::pow(col, glm::vec3(1.0f / 2.2f)); 
        frameBuffer[idx+0] = glm::clamp(col.x, 0.0f, 1.0f);
        frameBuffer[idx+1] = glm::clamp(col.y, 0.0f, 1.0f);
        frameBuffer[idx+2] = glm::clamp(col.z, 0.0f, 1.0f);
        return;
    }

    float t_current = fmaxf(inter.x, 0.0f) + params.stepSize * 0.5f; 
    float t_end = inter.y;
    float step_size = params.stepSize;

    glm::vec3 accumulatedColor(0.0f);
    float accumulated_transmittance = 1.0f;
    float current_rain_source_strength = 0.0f;
    const float ISO_PHASE = 1.0f/(4.0f*3.14159265f);


    while (t_current < t_end && accumulated_transmittance > 0.01f) { 
        glm::vec3 current_pos = rayOrig + rayDir * t_current;
        float baseDensity = sampleDensity(cloudDensityTexture, params.volumeMin, params.volumeDims, current_pos);

        float detailDensity = sampleDensity(detailNoiseTexture, params.volumeMin, params.volumeDims, current_pos);
        float cloud_density = fmaxf(baseDensity + detailDensity * params.detailWeight, 0.0f);

        
        if (baseDensity > 0.0001f) { 
            float extinction = cloud_density * params.extinctionScale;
            
            
            
            float current_segment_transmittance = __expf(-extinction * step_size);

            float cos_theta = glm::dot(rayDir, params.sunDirection); 
            float phase = henyeyGreenstein(cos_theta, params.anisotropy);
            float phaseMix   = glm::mix(phase, ISO_PHASE, 0.3f);
            
            float sun_transmittance_to_cloud_point = computeSunTransmittanceDetail(current_pos, cloudDensityTexture, detailNoiseTexture,params.sunDirection, params.shadowStep, params.shadowMaxDist, params);
            glm::vec3 sun_in_scattering = phaseMix * (params.sunColor * params.sunIntensity * sun_transmittance_to_cloud_point) *(1.0f - current_segment_transmittance);
            
            if (params.isRaining) {
                float luma = dot(sun_in_scattering, glm::vec3(0.299f, 0.587f, 0.114f));
                sun_in_scattering = glm::mix(sun_in_scattering, glm::vec3(luma), 0.3f); 
            }

            glm::vec3 multi_in_scattering = params.sunColor * params.sunIntensity * (1.0f - accumulated_transmittance) * params.multiScatterScale;
            accumulatedColor += multi_in_scattering * accumulated_transmittance;

                    glm::vec3 grad;
        {
            float dX = sampleDensity(cloudDensityTexture, params.volumeMin, params.volumeDims, current_pos + glm::vec3(step_size,0,0));
            float dY = sampleDensity(cloudDensityTexture, params.volumeMin, params.volumeDims, current_pos + glm::vec3(0,step_size,0));
            float dZ = sampleDensity(cloudDensityTexture, params.volumeMin, params.volumeDims, current_pos + glm::vec3(0,0,step_size));
            grad = glm::normalize(glm::vec3(dX,dY,dZ) - cloud_density + glm::vec3(1e-6f));
        }
            float Nup = max(glm::dot(grad, glm::vec3(0,1,0)), 0.0f);
            glm::vec3 skyDome = glm::mix(params.horizonColor, params.zenithColor, Nup);
            glm::vec3 ambIns  = skyDome * 0.5f * (1.0f - current_segment_transmittance);

            accumulatedColor += ambIns * accumulated_transmittance;
            accumulatedColor += sun_in_scattering * accumulated_transmittance;
            accumulated_transmittance *= current_segment_transmittance;
        }

       if (params.isRaining && accumulated_transmittance > 0.01f) {
            glm::vec3 rainLineColor = glm::vec3(0.75f, 0.8f, 0.85f); 
            // Base opacity for a visible line segment
            float rainLineBaseOpacity = 0.4f;

            // Sample rain density at the current ray position.
            glm::vec3 rain_sample_pos = current_pos;

            // Use integer parts of world coordinates for a more stable hash basis
            float cell_coord_x = floorf((current_pos.x - params.volumeMin.x) / params.cellSize.x);
            float cell_coord_z = floorf((current_pos.z - params.volumeMin.z) / params.cellSize.z);
            float h_x = fract(sinf(cell_coord_x * 7.31f + cell_coord_z * 11.47f + params.totalSimTime * 0.02f) * 13758.5f);
            float h_z = fract(cosf(cell_coord_z * 5.83f + cell_coord_x * 13.91f + params.totalSimTime * 0.015f) * 21783.1f);
            rain_sample_pos.x += (h_x - 0.5f) * params.cellSize.x * 0.15f;
            rain_sample_pos.z += (h_z - 0.5f) * params.cellSize.z * 0.15f;


            float density_at_sample_point = sampleDensity(rainDensityTexture, params.volumeMin, params.volumeDims, rain_sample_pos);

            // Threshold for considering this point part of a rain line. If density is > this, we are inside a rain packet.
            const float RAIN_LINE_DENSITY_THRESHOLD = 0.20f; 

            if (density_at_sample_point > RAIN_LINE_DENSITY_THRESHOLD) {
                // Modulate opacity slightly by how much density exceeds the threshold
                float opacity_factor = smoothstep(RAIN_LINE_DENSITY_THRESHOLD,
                                                  RAIN_LINE_DENSITY_THRESHOLD + 0.4f,
                                                  density_at_sample_point);

                float final_line_opacity = rainLineBaseOpacity * opacity_factor;

                accumulatedColor += rainLineColor * final_line_opacity * accumulated_transmittance;

                // Extinction from rain lines
                // Scale extinction by how "solid" this part of the line is.
                float effective_density_for_extinction = density_at_sample_point * opacity_factor;
                float rain_extinction_value = effective_density_for_extinction * params.rainExtinctionScale * 0.5f;
                float rain_transmittance_this_segment = __expf(-rain_extinction_value * step_size);
                accumulated_transmittance *= rain_transmittance_this_segment;
            }
        }
        
        t_current += step_size;
    }

    glm::vec3 skyColor = computeSkyColor(rayDir, params.zenithColor, params.horizonColor, params.sunDirection, params.sunColor, params.sunExponent, params.sunIntensity, params.isRaining);
    glm::vec3 finalColor = accumulatedColor + skyColor * accumulated_transmittance;

    finalColor = finalColor / (glm::vec3(1.0f)+ finalColor); 
    finalColor = glm::pow(finalColor, glm::vec3(1.0f / 2.2f));

    frameBuffer[idx] = glm::clamp(finalColor.x, 0.0f, 1.0f);
    frameBuffer[idx+1] = glm::clamp(finalColor.y, 0.0f, 1.0f);
    frameBuffer[idx+2] = glm::clamp(finalColor.z, 0.0f, 1.0f);
}

extern "C" {
    void initCuda(int imageWidth, int imageHeight, int grid_x, int grid_y, int grid_z);
    void cleanUpCuda();
    void computeFrame(float* hostFrameBuffer, float* hostCloudDensityField, float* hostRainDensityField, float* hostDetailNoiseField,const CudaParams& params);
}

void initCuda(int imageWidth, int imageHeight, int grid_x, int grid_y, int grid_z) {
    d_width_render = imageWidth;
    d_height_render = imageHeight;
    d_grid_x_render = grid_x;
    d_grid_y_render = grid_y;
    d_grid_z_render = grid_z;

    size_t bufferSize = (size_t)imageWidth * imageHeight * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_frameBuffer, bufferSize));
    std::cout << "Allocating frame buffer on GPU: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(grid_x, grid_y, grid_z);
    CUDA_CHECK(cudaMalloc3DArray(&d_densityArray, &channelDesc, extent, cudaArrayDefault));

    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_densityArray;

    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; 
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;     
    texDesc.readMode = cudaReadModeElementType;    
    texDesc.normalizedCoords = 1;                  

    CUDA_CHECK(cudaCreateTextureObject(&d_densityTexture, &resDesc, &texDesc, nullptr));
    
    size_t volumeSizeBytes = (size_t)grid_x * grid_y * grid_z * sizeof(float);
    std::cout << "Allocating 3D CUDA Array for density texture on GPU: " << volumeSizeBytes / (1024.0 * 1024.0) << " MB" << std::endl;


    CUDA_CHECK(cudaMalloc3DArray(&d_rainDensityArray, &channelDesc, extent, cudaArrayDefault));
    cudaResourceDesc rainResDesc;
    std::memset(&rainResDesc, 0, sizeof(rainResDesc));
    rainResDesc.resType = cudaResourceTypeArray;
    rainResDesc.res.array.array = d_rainDensityArray;
    cudaTextureDesc rainTexDesc;
    std::memset(&rainTexDesc, 0, sizeof(rainTexDesc));
    rainTexDesc.addressMode[0] = cudaAddressModeClamp;
    rainTexDesc.addressMode[1] = cudaAddressModeClamp;
    rainTexDesc.addressMode[2] = cudaAddressModeClamp;
    rainTexDesc.filterMode = cudaFilterModeLinear;
    rainTexDesc.readMode = cudaReadModeElementType;
    rainTexDesc.normalizedCoords = 1;
    CUDA_CHECK(cudaCreateTextureObject(&d_rainDensityTexture, &rainResDesc, &rainTexDesc, nullptr));

    // Detail noise texture
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>(); 
    cudaExtent extent3D = make_cudaExtent(grid_x*2, grid_y*2, grid_z*2);
    CUDA_CHECK(cudaMalloc3DArray(&d_detailNoiseArray, &channelDescFloat, extent3D, cudaArrayDefault));

    cudaResourceDesc resDescDetail;
    std::memset(&resDescDetail, 0, sizeof(resDescDetail));
    resDescDetail.resType = cudaResourceTypeArray;
    resDescDetail.res.array.array = d_detailNoiseArray;

    cudaTextureDesc texDescDetail;
    std::memset(&texDescDetail, 0, sizeof(texDescDetail));
    texDescDetail.addressMode[0] = cudaAddressModeWrap;
    texDescDetail.addressMode[1] = cudaAddressModeWrap;
    texDescDetail.addressMode[2] = cudaAddressModeWrap;
    texDescDetail.filterMode = cudaFilterModeLinear;
    texDescDetail.readMode = cudaReadModeElementType;
    texDescDetail.normalizedCoords = 1;
    CUDA_CHECK(cudaCreateTextureObject(&d_detailNoiseTexture, &resDescDetail, &texDescDetail, nullptr));

    std::cout << "CUDA Initialized Successfully." << std::endl;
}

void cleanUpCuda() {
    if (d_frameBuffer) {
        CUDA_CHECK(cudaFree(d_frameBuffer));
        d_frameBuffer = nullptr;
    }
    if (d_densityTexture != 0) {
        CUDA_CHECK(cudaDestroyTextureObject(d_densityTexture));
        d_densityTexture = 0;
    }
    if (d_densityArray != nullptr) {
        CUDA_CHECK(cudaFreeArray(d_densityArray));
        d_densityArray = nullptr;
    }

    if (d_rainDensityTexture != 0) {
    CUDA_CHECK(cudaDestroyTextureObject(d_rainDensityTexture));
    d_rainDensityTexture = 0;
    }
    if (d_rainDensityArray != nullptr) {
        CUDA_CHECK(cudaFreeArray(d_rainDensityArray));
        d_rainDensityArray = nullptr;
    }

     if (d_detailNoiseTexture != 0) {
        CUDA_CHECK(cudaDestroyTextureObject(d_detailNoiseTexture));
        d_detailNoiseTexture = 0;
    }
    if (d_detailNoiseArray != nullptr) {
        CUDA_CHECK(cudaFreeArray(d_detailNoiseArray));
        d_detailNoiseArray = nullptr;
    }

    std::cout << "CUDA Cleaned Up." << std::endl;
}

void computeFrame(float* hostFrameBuffer, float* hostCloudDensityField, float* hostRainDensityField, float* hostDetailNoiseField,const CudaParams& params) {

    // This data goes into d_densityArray, which is associated with d_densityTexture (for clouds)
    cudaMemcpy3DParms cloudCopyParams;
    memset(&cloudCopyParams, 0, sizeof(cloudCopyParams));

    // Source is now hostCloudDensityField
    cloudCopyParams.srcPtr = make_cudaPitchedPtr((void*)hostCloudDensityField, d_grid_x_render * sizeof(float), d_grid_x_render, d_grid_y_render);
    cloudCopyParams.dstArray = d_densityArray;
    cloudCopyParams.dstPos = make_cudaPos(0, 0, 0);
    cudaExtent copyExtent = make_cudaExtent(d_grid_x_render, d_grid_y_render, d_grid_z_render);
    cloudCopyParams.extent = copyExtent;
    cloudCopyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&cloudCopyParams));

    cudaMemcpy3DParms rainCopyParams;
    memset(&rainCopyParams, 0, sizeof(rainCopyParams));

    // Source is hostRainDensityField
    rainCopyParams.srcPtr = make_cudaPitchedPtr((void*)hostRainDensityField, d_grid_x_render * sizeof(float), d_grid_x_render, d_grid_y_render);
    rainCopyParams.dstArray = d_rainDensityArray;
    rainCopyParams.dstPos = make_cudaPos(0, 0, 0);
    rainCopyParams.extent = copyExtent;
    rainCopyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&rainCopyParams));

    // Copy detail noise
    cudaMemcpy3DParms detailNoiseCopyParams;
    memset(&detailNoiseCopyParams, 0, sizeof(detailNoiseCopyParams));
    detailNoiseCopyParams.srcPtr = make_cudaPitchedPtr((void*)hostDetailNoiseField, d_grid_x_render *2* sizeof(float), d_grid_x_render*2, d_grid_y_render*2);
    detailNoiseCopyParams.dstArray = d_detailNoiseArray;
    detailNoiseCopyParams.dstPos = make_cudaPos(0, 0, 0);
    cudaExtent copyExtentNoise = make_cudaExtent(d_grid_x_render, d_grid_y_render, d_grid_z_render);
    detailNoiseCopyParams.extent = copyExtentNoise;
    detailNoiseCopyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&detailNoiseCopyParams));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (d_width_render + blockSize.x - 1) / blockSize.x,
        (d_height_render + blockSize.y - 1) / blockSize.y
    );

    rayTraceKernel<<<gridSize, blockSize>>>(d_frameBuffer, d_densityTexture, d_rainDensityTexture, d_detailNoiseTexture, params);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t bufferSize = (size_t)d_width_render * d_height_render * 3 * sizeof(float);
    CUDA_CHECK(cudaMemcpy(hostFrameBuffer, d_frameBuffer, bufferSize, cudaMemcpyDeviceToHost));
}