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
#include <cmath>


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define IX(i,j,k)  ( ((k)*height + (j))*width + (i) )

struct CudaSimulationParams {
    float DT = 0.05f;
    float DENSITY_DIFFUSION;
    float VAPOR_DIFFUSION;
    float TEMP_DIFFUSION = 0.00002f;
    float VISCOSITY = 0.000001f;
    int SOLVER_ITERATIONS = 2;
    float BACKGROUND_WIND_X = 0.01f;
    float QC_DISSIPATION = 0.9995f;
    float QV_DISSIPATION = 1.0f;
    float TEMP_DISSIPATION = 1.0f;
    float BUOYANCY_ALPHA = 0.08f;
    float BUOYANCY_BETA = 0.12f;
    float BASE_AIR_DENSITY_FOR_BUOYANCY = 1.225f;
    float GRAVITY_ACCEL = 9.81f;
    float AMBIENT_TEMP_SEA_LEVEL_K = 288.15f;
    float AMBIENT_LAPSE_RATE_K_PER_M = 0.0065f;
    float AMBIENT_VAPOR_SEA_LEVEL = 0.008f;
    float AMBIENT_VAPOR_DECREASE_RATE = 0.000002f;
    float CONDENSATION_RATE = 0.1f;
    float EVAPORATION_RATE = 0.01f;
    float LATENT_HEAT_VAPORIZATION = 2.501e6f;
    float SPECIFIC_HEAT_AIR_PRESSURE = 1005.0f;
    float GAS_CONSTANT_DRY_AIR = 287.058f;
    float GAS_CONSTANT_WATER_VAPOR = 461.5f;
    float CP_OVER_RD = SPECIFIC_HEAT_AIR_PRESSURE / GAS_CONSTANT_DRY_AIR;
    float EPSILON_RD_OVER_RV = GAS_CONSTANT_DRY_AIR / GAS_CONSTANT_WATER_VAPOR;
    float CELL_DY;
};



int d_grid_x = 0;
int d_grid_y = 0;
int d_grid_z = 0;

int d_width = 0;
int d_height = 0;
int d_depth = 0;
int d_grid_size = 0;
float* d_qcVolumeData = nullptr;
float* d_Vx = nullptr;
float* d_Vy = nullptr;
float* d_Vz = nullptr;
float* d_Vx_prev = nullptr;
float* d_Vy_prev = nullptr;
float* d_Vz_prev = nullptr;
float* d_temperature = nullptr;
float* d_temperature_prev = nullptr;
float* d_water_vapor = nullptr;
float* d_qv_prev = nullptr;
float* d_condensed_water = nullptr;
float* d_qc_prev = nullptr;

extern "C" {
    void initGpuSimulation(int grid_x, int grid_y, int grid_z);
    void cleanupSimulationCuda();
    void updateSimGPU(float* Vx, float* Vy, float* Vz, float* Vx_prev, float* Vy_prev, float* Vz_prev,
        float* temperature, float* water_vapor, float* condensed_water, float* temp_prev, float* qv_prev, float* qc_prev,  CudaSimulationParams& params
    );
}

void initGpuSimulation(int grid_x, int grid_y, int grid_z) {
    d_grid_x = grid_x;
    d_grid_y = grid_y;
    d_grid_z = grid_z;
    d_width = grid_x + 2;
    d_height = grid_y + 2;
    d_depth = grid_z + 2;
    d_grid_size = d_width * d_height * d_depth;

    size_t bufferSize = d_grid_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Vx, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_Vy, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_Vz, bufferSize));

    CUDA_CHECK(cudaMalloc(&d_Vx_prev, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_Vy_prev, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_Vz_prev, bufferSize));

    CUDA_CHECK(cudaMalloc(&d_temperature, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_temperature_prev, bufferSize));

    CUDA_CHECK(cudaMalloc(&d_water_vapor, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_qv_prev, bufferSize));

    CUDA_CHECK(cudaMalloc(&d_condensed_water, bufferSize));
    CUDA_CHECK(cudaMalloc(&d_qc_prev, bufferSize));

    // Qc doesnt have boundaries
    size_t qcBufferSize = d_grid_x * d_grid_y * d_grid_z * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_qcVolumeData, qcBufferSize));
    CUDA_CHECK(cudaMemset(d_Vx, 0, bufferSize));


    std::cout << "Allocating simulation data on GPU: " << (bufferSize * 12 + qcBufferSize) / (1024.0 * 1024.0) << " MB" << std::endl;

}

void cleanupSimulationCuda() {

    if (d_Vx) {
        CUDA_CHECK(cudaFree(d_Vx));
        d_Vx = nullptr;
    }
    if (d_Vy) {
        CUDA_CHECK(cudaFree(d_Vy));
        d_Vy = nullptr;
    }
    if (d_Vz) {
        CUDA_CHECK(cudaFree(d_Vz));
        d_Vz = nullptr;
    }
    if (d_Vx_prev) {
        CUDA_CHECK(cudaFree(d_Vx_prev));
        d_Vx_prev = nullptr;
    }
    if (d_Vy_prev) {
        CUDA_CHECK(cudaFree(d_Vy_prev));
        d_Vy_prev = nullptr;
    }
    if (d_Vz_prev) {
        CUDA_CHECK(cudaFree(d_Vz_prev));
        d_Vz_prev = nullptr;
    }
    if (d_temperature) {
        CUDA_CHECK(cudaFree(d_temperature));
        d_temperature = nullptr;
    }
    if (d_temperature_prev) {
        CUDA_CHECK(cudaFree(d_temperature_prev));
        d_temperature_prev = nullptr;
    }
    if (d_water_vapor) {
        CUDA_CHECK(cudaFree(d_water_vapor));
        d_water_vapor = nullptr;
    }
    if (d_qv_prev) {
        CUDA_CHECK(cudaFree(d_qv_prev));
        d_qv_prev = nullptr;
    }
    if (d_condensed_water) {
        CUDA_CHECK(cudaFree(d_condensed_water));
        d_condensed_water = nullptr;
    }
    if (d_qc_prev) {
        CUDA_CHECK(cudaFree(d_qc_prev));
        d_qc_prev = nullptr;
    }
    if (d_qcVolumeData) {
        CUDA_CHECK(cudaFree(d_qcVolumeData));
        d_qcVolumeData = nullptr;
    }

    printf("Cleaned up simulation variables \n");
}


// Set boundary =========================
__global__
void _setBndFaces(int b, float* x, int width,int height,int depth, bool openX, bool openY, bool openZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= width || j>= height || k >= depth) return;

    // only act on face‐pixels
    bool onXface = (i==0 || i==width-1) && (j>0 && j<height-1) && (k>0 && k<depth-1);
    bool onYface = (j==0 || j==height-1) && (i>0 && i<width-1)  && (k>0 && k<depth-1);
    bool onZface = (k==0 || k==depth-1) && (i>0 && i<width-1)   && (j>0 && j<height-1);

    if (!(onXface||onYface||onZface)) return;

    int ii = i, jj = j, kk = k;
    // choose neighbor inside the volume
    if (onXface) {
      ii = (i==0 ? 1 : width-2);
    } else if (onYface) {
      jj = (j==0 ? 1 : height-2);
    } else { // onZface
      kk = (k==0 ? 1 : depth-2);
    }

    float v = x[IX(ii,jj,kk)];
    // reflect only when b matches the axis (1=X, 2=Y, 3=Z)
    if ((b==1 && onXface && !openX) || (b==2 && onYface && !openY) || (b==3 && onZface && !openZ)) {
        v = -v;
    }
    x[IX(i,j,k)] = v;
}
__global__
void _setBndEdges(int b, float* x, int width,int height,int depth)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= width || j>= height || k >= depth) return;

    // edges/corners: any coord at 0 or max along >= 2
    bool atI = (i==0 || i==width-1),
         atJ = (j==0 || j==height-1),
         atK = (k==0 || k==depth-1);
    int count = (atI?1:0)+(atJ?1:0)+(atK?1:0);
    if (count<2) return;  // not an edge or corner

    // gather neighbors: average over the inner‐facing neighbors
    float sum = 0.0f;
    int n   = 0;
    // Check bounds before accessing neighbors if i,j,k could be exactly on the boundary
    float neighbor_val;
    if (i==0) { 
        neighbor_val = x[IX(1, j, k)]; 
        sum += neighbor_val; 
        n++; 
    }
    else if(i==width-1) { 
        neighbor_val = x[IX(width-2, j, k)]; 
        sum += neighbor_val; 
        n++; 
    }

    if (j==0) { neighbor_val = x[IX(i, 1, k)]; 
        sum += neighbor_val; 
        n++; 
    } else if(j==height-1) { 
        neighbor_val = x[IX(i, height-2, k)]; 
        sum += neighbor_val; 
        n++; 
    }

    if (k==0) { neighbor_val = x[IX(i, j, 1)]; 
        sum += neighbor_val; 
        n++; 
    } else if(k==depth-1) { 
        neighbor_val = x[IX(i, j, depth-2)]; 
        sum += neighbor_val; 
        n++; }
    if (n > 0) {
        x[IX(i,j,k)] = sum / float(n);
    } else {
        x[IX(i,j,k)] = 0.0f;
    }
}

// Linear Solve ===============================
__global__
void _jacobiIter(int b, float* x, const float* x0, float a, float cRecip, int width,int height,int depth)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    // skip boundaries entirely
    if (i==0||i>=width-1 || j==0||j>=height-1 || k==0||k>=depth-1)
      return;

    int idx = IX(i,j,k);
    float sum =
         x[IX(i+1,j,k)] + x[IX(i-1,j,k)]
       + x[IX(i,j+1,k)] + x[IX(i,j-1,k)]
       + x[IX(i,j,k+1)] + x[IX(i,j,k-1)];


    float newVal = (x0[idx] + a*sum) * cRecip;
    if (isnan(newVal) || isinf(newVal)) {
        newVal = x0[idx];
    }
    x[idx] = newVal;
}

// Pressure Projection ==========================
__global__ void computeDivergenceKernel(float* div, const float* Vx, const float* Vy, const float* Vz, int width, int height, int depth, float scale) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i<1||i>=width-1 || j<1||j>=height-1 || k<1||k>=depth-1) 
        return;

    int idx = IX(i,j,k);
    div[idx] = -0.5f * (
         Vx[IX(i+1,j,k)] - Vx[IX(i-1,j,k)]
       + Vy[IX(i,j+1,k)] - Vy[IX(i,j-1,k)]
       + Vz[IX(i,j,k+1)] - Vz[IX(i,j,k-1)]
      );
}
__global__ void subtractPressureGradientKernel(float* Vx, float* Vy, float* Vz, const float* p, int width, int height, int depth, float scale) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i<1||i>=width-1 || j<1||j>=height-1 || k<1||k>=depth-1) 
        return;

    int idx = IX(i,j,k);
    Vx[idx] -= 0.5f * (p[IX(i+1,j,k)] - p[IX(i-1,j,k)]);
    Vy[idx] -= 0.5f * (p[IX(i,j+1,k)] - p[IX(i,j-1,k)]);
    Vz[idx] -= 0.5f * (p[IX(i,j,k+1)] - p[IX(i,j,k-1)]);
}

void projectGPU(int iter, float cell_dy, const dim3& gridPadded_dim, const dim3& gridInner_dim, const dim3& block_dim) {
    size_t bufferSize = d_grid_size  * sizeof(float);
    // Compute divergence
    dim3 block(8,8,8); 
    dim3 grid(
        (d_width  + block.x-1)/block.x,
        (d_height + block.y-1)/block.y,
        (d_depth  + block.z-1)/block.z
    );
    // use previous y velocity grid as the divergence storage

    // Use scaling 
    float scale = 1.0f / (cell_dy * 2.0f);
    computeDivergenceKernel<<<grid,block>>>(d_Vy_prev, d_Vx, d_Vy, d_Vz, d_width, d_height, d_depth, scale);
    cudaDeviceSynchronize();

    float neg_sq_scale = -scale * scale;

    _setBndFaces<<<grid,block>>>(0, d_Vy_prev, d_width, d_height, d_depth, false, false, false);
    cudaDeviceSynchronize();

    _setBndEdges<<<grid,block>>>(0, d_Vy_prev, d_width, d_height, d_depth);
    cudaDeviceSynchronize();

    // Zero out pressure and solve
    CUDA_CHECK(cudaMemset(d_Vx_prev, 0, bufferSize));

    _setBndFaces<<<grid,block>>>(0, d_Vx_prev, d_width, d_height, d_depth, false, false, false);
    cudaDeviceSynchronize();

    _setBndEdges<<<grid,block>>>(0, d_Vx_prev, d_width, d_height, d_depth);
    cudaDeviceSynchronize();

    for (int it=0; it<iter; ++it) {
        _jacobiIter<<<grid,block>>>(0, d_Vx_prev, d_Vy_prev, 1.0f, 1.0f/6.0f,
                                    d_width, d_height, d_depth);
        cudaDeviceSynchronize();
        _setBndFaces<<<grid,block>>>(0, d_Vx_prev, d_width, d_height, d_depth, false, false, false);
        cudaDeviceSynchronize();
        _setBndEdges<<<grid,block>>>(0, d_Vx_prev, d_width, d_height, d_depth);
        cudaDeviceSynchronize();
    }

    // Subtract gradient
    subtractPressureGradientKernel<<<grid,block>>>(d_Vx, d_Vy, d_Vz,d_Vx_prev, d_width, d_height, d_depth, scale);
    cudaDeviceSynchronize();

    _setBndFaces<<<grid,block>>>(1, d_Vx, d_width, d_height, d_depth, true, false, false);
    cudaDeviceSynchronize();
    _setBndEdges<<<grid,block>>>(1, d_Vx, d_width, d_height, d_depth);
    cudaDeviceSynchronize();
    _setBndFaces<<<grid,block>>>(2, d_Vy, d_width, d_height, d_depth, false, true, false);
    cudaDeviceSynchronize();
    _setBndEdges<<<grid,block>>>(2, d_Vy, d_width, d_height, d_depth);
    cudaDeviceSynchronize();
    _setBndFaces<<<grid,block>>>(3, d_Vz, d_width, d_height, d_depth, false, false, true);
    cudaDeviceSynchronize();
    _setBndEdges<<<grid,block>>>(3, d_Vz, d_width, d_height, d_depth);
    cudaDeviceSynchronize();
}


// ====================================================

// Advection =======================
__global__ void advectKernel(int b, float* d, const float* d0, const float* velX, const float* velY, const float* velZ, int width,  int height,  int depth, float dt, float cell_dy) {
    // Compute global 3D indices
    int i_inner = blockIdx.x * blockDim.x + threadIdx.x;
    int j_inner = blockIdx.y * blockDim.y + threadIdx.y;
    int k_inner = blockIdx.z * blockDim.z + threadIdx.z;

    int i = i_inner + 1;
    int j = j_inner + 1;
    int k = k_inner + 1;

    if (i >= width-1 || j >= height-1 || k >= depth-1) return;

    int idx = IX(i,j,k);

    float displacement_multiplier = dt /cell_dy;

    // Backtrace
    float x = (float)i - displacement_multiplier* velX[idx];
    float y = (float)j - displacement_multiplier * velY[idx];
    float z = (float)k - displacement_multiplier * velZ[idx];

    // Clamp
    x = fminf(fmaxf(x, 0.5f), (float)width - 1.5f);
    y = fminf(fmaxf(y, 0.5f), (float)height - 1.5f);
    z = fminf(fmaxf(z, 0.5f), (float)depth - 1.5f);

    // Floor indices
    int i0 = floorf(x); int i1 = i0 + 1;
    int j0 = floorf(y); int j1 = j0 + 1;
    int k0 = floorf(z); int k1 = k0 + 1;

    // Interpolation weights
    float s1 = x - (float)i0, s0 = 1.0f - s1;
    float t1 = y - (float)j0, t0 = 1.0f - t1;
    float r1 = z - (float)k0, r0 = 1.0f - r1;


    // Fetch corners
    float d000 = d0[IX(i0,j0,k0)];
    float d100 = d0[IX(i1,j0,k0)];
    float d010 = d0[IX(i0,j1,k0)];
    float d110 = d0[IX(i1,j1,k0)];
    float d001 = d0[IX(i0,j0,k1)];
    float d101 = d0[IX(i1,j0,k1)];
    float d011 = d0[IX(i0,j1,k1)];
    float d111 = d0[IX(i1,j1,k1)];

    // Trilinear interp
    float d_j0k0 = s0*d000 + s1*d100;
    float d_j1k0 = s0*d010 + s1*d110;
    float d_j0k1 = s0*d001 + s1*d101;
    float d_j1k1 = s0*d011 + s1*d111;

    float d_k0 = t0*d_j0k0 + t1*d_j1k0;
    float d_k1 = t0*d_j0k1 + t1*d_j1k1;

    float val = r0*d_k0 + r1*d_k1;
    d[idx] = isnan(val) ? 0.0f : val;
}

__global__ void advectQCKernel(float* d_qc, const float* d_qc0, const float* velX, const float* velY, const float* velZ, int width, int height, int depth, float dt, float cell_dy, float sedimentation_speed) {
    int i_inner = blockIdx.x * blockDim.x + threadIdx.x;
    int j_inner = blockIdx.y * blockDim.y + threadIdx.y;
    int k_inner = blockIdx.z * blockDim.z + threadIdx.z;

    // Convert inner index (0..N-1) to padded index (1..N)
    int i = i_inner + 1;
    int j = j_inner + 1;
    int k = k_inner + 1;

    // Check if thread is outside the inner grid bounds
    if (i >= width - 1 || j >= height - 1 || k >= depth - 1) return;

    // Calculate scaling factors
    // float dtx = dt * (float)(width - 2);
    // float dty = dt * (float)(height - 2);
    // float dtz = dt * (float)(depth - 2);

    float displacement_multiplier = dt / cell_dy;

    int idx = IX(i, j, k);

    // Subtract sedimentation speed (positive value represents downward movement) from the fluid's vertical velocity component.
    float effective_vy = velY[idx] - sedimentation_speed;

    // Backtrace particle origin position using effective Y velocity
    float x = (float)i - displacement_multiplier * velX[idx];
    float y = (float)j - displacement_multiplier * effective_vy;
    float z = (float)k - displacement_multiplier * velZ[idx];

    // Clamp the backtraced position (same as standard advection)
    x = fminf(fmaxf(x, 0.5f), (float)width - 1.5f);
    y = fminf(fmaxf(y, 0.5f), (float)height - 1.5f);
    z = fminf(fmaxf(z, 0.5f), (float)depth - 1.5f);

    // Get integer indices of the corners
    int i0 = floorf(x); int i1 = i0 + 1;
    int j0 = floorf(y); int j1 = j0 + 1;
    int k0 = floorf(z); int k1 = k0 + 1;

    // Calculate interpolation weights
    float s1 = x - (float)i0; float s0 = 1.0f - s1;
    float t1 = y - (float)j0; float t0 = 1.0f - t1;
    float r1 = z - (float)k0; float r0 = 1.0f - r1;

    // Fetch QC values from the previous timestep (d_qc0) at the 8 corners
    float qc000 = d_qc0[IX(i0, j0, k0)]; float qc100 = d_qc0[IX(i1, j0, k0)];
    float qc010 = d_qc0[IX(i0, j1, k0)]; float qc110 = d_qc0[IX(i1, j1, k0)];
    float qc001 = d_qc0[IX(i0, j0, k1)]; float qc101 = d_qc0[IX(i1, j0, k1)];
    float qc011 = d_qc0[IX(i0, j1, k1)]; float qc111 = d_qc0[IX(i1, j1, k1)];

    // Perform trilinear interpolation
    float qc_j0k0 = s0 * qc000 + s1 * qc100;
    float qc_j1k0 = s0 * qc010 + s1 * qc110;
    float qc_j0k1 = s0 * qc001 + s1 * qc101;
    float qc_j1k1 = s0 * qc011 + s1 * qc111;
    float qc_k0 = t0 * qc_j0k0 + t1 * qc_j1k0;
    float qc_k1 = t0 * qc_j0k1 + t1 * qc_j1k1;
    float interp_val = r0 * qc_k0 + r1 * qc_k1;

    // Write the interpolated value to the current timestep QC array (d_qc)
    d_qc[idx] = isnan(interp_val) ? 0.0f : interp_val;
}
// ===================

// Thermodynamics =======================

__device__ __forceinline__ float calculate_saturation_vapor_pressure_gpu(float T_kelvin) {
    if (T_kelvin <= 0.0f) return 0.0f;
    float T_celsius = T_kelvin - 273.15f;
    return 611.2f * expf((17.67f * T_celsius) / (T_celsius + 243.5f));
}

__device__ __forceinline__ float calculate_saturation_mixing_ratio_gpu(float T_kelvin, float p_pascal, const CudaSimulationParams& params) {
    if (p_pascal <= 0.0f) return 0.0f;
    float es = calculate_saturation_vapor_pressure_gpu(T_kelvin);
    float denom = p_pascal - es;
    if (denom <= 1e-6f) return 1.0f;
    return fmaxf(0.0f, (params.EPSILON_RD_OVER_RV * es) / denom);
}

__device__ __forceinline__ float get_ambient_temperature_gpu(int j_index, const CudaSimulationParams& params) {
    float height_m = (float)(j_index - 1 + 0.5f) * params.CELL_DY;
    height_m = fmaxf(0.0f, height_m);
    return params.AMBIENT_TEMP_SEA_LEVEL_K - params.AMBIENT_LAPSE_RATE_K_PER_M * height_m;
}

__device__ __forceinline__ float get_ambient_vapor_gpu(int j_index, const CudaSimulationParams& params) {
    float height_m = (float)(j_index - 1 + 0.5f) * params.CELL_DY;
    height_m = fmaxf(0.0f, height_m);
    return fmaxf(0.0f, params.AMBIENT_VAPOR_SEA_LEVEL - params.AMBIENT_VAPOR_DECREASE_RATE * height_m);
}

__device__ __forceinline__ float estimate_pressure_gpu(int j_index, const CudaSimulationParams& params) {
    float height_m = (float)(j_index - 1 + 0.5f) * params.CELL_DY;
    height_m = fmaxf(0.0f, height_m);
    const float scale_height = 8000.0f;
    return 101325.0f * expf(-height_m / scale_height);
}

__global__ void updateThermoPhaseKernel(float* __restrict__ temperature, float* __restrict__ water_vapor, float* __restrict__ condensed_water, int width, int height, int depth, const CudaSimulationParams params) {
    int i_inner = blockIdx.x * blockDim.x + threadIdx.x;
    int j_inner = blockIdx.y * blockDim.y + threadIdx.y;
    int k_inner = blockIdx.z * blockDim.z + threadIdx.z;

    int i = i_inner + 1;
    int j = j_inner + 1;
    int k = k_inner + 1;

    if (i >= width - 1 || j >= height - 1 || k >= depth - 1) return;

    int idx = IX(i, j, k);

    float T_k = temperature[idx];
    float qv = water_vapor[idx];
    float qc = condensed_water[idx];

    float p_pascal = estimate_pressure_gpu(j, params);
    float qs = calculate_saturation_mixing_ratio_gpu(T_k, p_pascal, params);
    float delta_qc = 0.0f;

    if (qs <= 0.0f) { qs = 1e-9f; }

    if (qv > qs) {
        float supersaturation = qv - qs;
        float amount_condensed = fminf(supersaturation, qv) * params.CONDENSATION_RATE;
        amount_condensed = fmaxf(0.0f, amount_condensed);
        delta_qc = amount_condensed;
        qv = fmaxf(0.0f, qv - delta_qc);
        qc += delta_qc;
    } else if (qc > 1e-9f) {
        float subsaturation = qs - qv;
        float amount_evaporated = fminf(fmaxf(0.0f, subsaturation), qc) * params.EVAPORATION_RATE;
        amount_evaporated = fmaxf(0.0f, amount_evaporated);
        delta_qc = -amount_evaporated;
        qv += amount_evaporated;
        qc = fmaxf(0.0f, qc + delta_qc);
    }

    if (fabsf(delta_qc) > 1e-12f) {
        T_k += (params.LATENT_HEAT_VAPORIZATION / params.SPECIFIC_HEAT_AIR_PRESSURE) * delta_qc;
        T_k = fminf(fmaxf(T_k, 150.0f), 350.0f);
    }

    qv = fmaxf(0.0f, qv);
    qc = fmaxf(0.0f, qc);

    if (isnan(T_k) || isinf(T_k) || isnan(qv) || isinf(qv) || isnan(qc) || isinf(qc)) {
        T_k = get_ambient_temperature_gpu(j, params);
        qv = get_ambient_vapor_gpu(j, params);
        qc = 0.0f;
    }

    temperature[idx] = T_k;
    water_vapor[idx] = qv;
    condensed_water[idx] = qc;
}




// ========================



void updateSimGPU(float* Vx, float* Vy, float* Vz, float* Vx_prev, float* Vy_prev, float* Vz_prev,
    float* temperature, float* water_vapor, float* condensed_water, float* temp_prev, float* qv_prev, float* qc_prev,  CudaSimulationParams& params
) {

    // First copy memory from CPU to GPU
    size_t bufferSize = d_grid_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_Vx, Vx, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vy, Vy, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vz, Vz, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vx_prev, Vx_prev, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vy_prev, Vy_prev, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vz_prev, Vz_prev, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temperature, temperature, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temperature_prev, temp_prev, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_water_vapor, water_vapor, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qv_prev, qv_prev, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_condensed_water, condensed_water, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qc_prev, qc_prev, bufferSize, cudaMemcpyHostToDevice));


    // Construct grid and block sizes
    dim3 block(8,8,8);
    // For boundary kernels
    dim3 gridPadded(
        (d_width  + block.x-1)/block.x,
        (d_height + block.y-1)/block.y,
        (d_depth  + block.z-1)/block.z
      );

    // For advection/computation kernels
    dim3 gridInner(
        (d_grid_x + block.x-1)/block.x, // d_grid_x = d_width - 2
        (d_grid_y + block.y-1)/block.y, // d_grid_y = d_height - 2
        (d_grid_z + block.z-1)/block.z  // d_grid_z = d_depth - 2
      );

    // lambda for diffusion
    auto diffuseField = [&](int b, float* d_x, float* d_x0, float diffRate) {
        // Swap buffers
        // Swap d_x and d_x0
        CUDA_CHECK(cudaMemcpy(d_x0, d_x, bufferSize, cudaMemcpyDeviceToDevice));
        
        // Compute linsolve coefficients
        // float a = params.DT * diffRate * d_grid_x * d_grid_y * d_grid_z;
        float a = params.DT * diffRate / (params.CELL_DY * params.CELL_DY);
        a = fminf(0.1f, a);
        float cRecip = 1.0f / (1.0f + 6.0f * a);

        // Iterations
        for (int it=0; it<params.SOLVER_ITERATIONS; ++it) {
            _jacobiIter<<<gridPadded,block>>>(b, d_x, d_x0, a, cRecip, d_width,d_height,d_depth);
            cudaDeviceSynchronize();
            // True to allow open x boundary condition
            _setBndFaces<<<gridPadded,block>>>(b, d_x, d_width,d_height,d_depth, true, true, true);
            cudaDeviceSynchronize();
            _setBndEdges<<<gridPadded,block>>>(b, d_x, d_width,d_height,d_depth);
            cudaDeviceSynchronize();
          }

    };

    // Diffuse the velocities
    diffuseField(1, d_Vx, d_Vx_prev, params.VISCOSITY);
    diffuseField(2, d_Vy, d_Vy_prev, params.VISCOSITY);
    diffuseField(3, d_Vz, d_Vz_prev, params.VISCOSITY);

    // Now project velocities (pressure correction). We will use vx_prev, and vy_prev as the pressure field, to reuse memory. 
    //These will be swapped later for advection (then vx_prev will hold the velocity field).
    projectGPU(params.SOLVER_ITERATIONS, params.CELL_DY, gridPadded, gridInner, block);

    // Now swap v and v prev
    CUDA_CHECK(cudaMemcpy(d_Vx_prev, d_Vx, bufferSize, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vy_prev, d_Vy, bufferSize, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vz_prev, d_Vz, bufferSize, cudaMemcpyDeviceToDevice));

    // Launch velocity advection kernels
    advectKernel<<<gridInner, block>>>(1, d_Vx, d_Vx_prev, d_Vx_prev, d_Vy_prev, d_Vz_prev, d_width, d_height, d_depth, params.DT, params.CELL_DY);
    CUDA_CHECK(cudaGetLastError());
    advectKernel<<<gridInner, block>>>(2, d_Vy, d_Vy_prev, d_Vx_prev, d_Vy_prev, d_Vz_prev, d_width, d_height, d_depth, params.DT, params.CELL_DY);
    CUDA_CHECK(cudaGetLastError());
    advectKernel<<<gridInner, block>>>(3, d_Vz, d_Vz_prev, d_Vx_prev, d_Vy_prev, d_Vz_prev, d_width, d_height, d_depth, params.DT, params.CELL_DY);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Set boundaries for velocity fields after advection
    _setBndFaces<<<gridPadded,block>>>(1, d_Vx, d_width, d_height, d_depth, true, false, false); // b=1, openX=true
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(1, d_Vx, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();

    _setBndFaces<<<gridPadded,block>>>(2, d_Vy, d_width, d_height, d_depth, false, true, false);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(2, d_Vy, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();

    _setBndFaces<<<gridPadded,block>>>(3, d_Vz, d_width, d_height, d_depth, false, false, true);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(3, d_Vz, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();

    // Results are now in Vx, Vy, Vz
    // At this point, could project velocity again if required.

    // Diffuse temperature, qv, and qc now
    diffuseField(0, d_temperature, d_temperature_prev, params.TEMP_DIFFUSION);
    diffuseField(0, d_water_vapor, d_qv_prev, params.VAPOR_DIFFUSION);
    diffuseField(0, d_condensed_water, d_qc_prev, params.DENSITY_DIFFUSION);
    printf("Using density diffusion of %f", params.DENSITY_DIFFUSION);

    // Advect temperature and water vapor
    CUDA_CHECK(cudaMemcpy(d_temperature_prev, d_temperature, bufferSize, cudaMemcpyDeviceToDevice));
    advectKernel<<<gridInner, block>>>(0, d_temperature, d_temperature_prev, d_Vx, d_Vy, d_Vz, d_width, d_height, d_depth, params.DT, params.CELL_DY);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize(); 
    _setBndFaces<<<gridPadded,block>>>(0, d_temperature, d_width, d_height, d_depth, true, true, true);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(0, d_temperature, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();


    // water vapor
    CUDA_CHECK(cudaMemcpy(d_qv_prev, d_water_vapor, bufferSize, cudaMemcpyDeviceToDevice));
    advectKernel<<<gridInner, block>>>(0, d_water_vapor, d_qv_prev, d_Vx, d_Vy, d_Vz, d_width, d_height, d_depth, params.DT, params.CELL_DY);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    _setBndFaces<<<gridPadded,block>>>(0, d_water_vapor, d_width, d_height, d_depth, true, true, true);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(0, d_water_vapor, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();

    // Condensed water
    CUDA_CHECK(cudaMemcpy(d_qc_prev, d_condensed_water, bufferSize, cudaMemcpyDeviceToDevice));
    float sedimentationSpeed = 0.01f;
    advectQCKernel<<<gridInner, block>>>(d_condensed_water, d_qc_prev, d_Vx, d_Vy, d_Vz, d_width, d_height, d_depth, params.DT, params.CELL_DY, sedimentationSpeed);
    cudaDeviceSynchronize();
    // Set bounds
    _setBndFaces<<<gridPadded,block>>>(0, d_condensed_water, d_width, d_height, d_depth, true, true, true);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    _setBndEdges<<<gridPadded,block>>>(0, d_condensed_water, d_width, d_height, d_depth);
    CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();   
    
    updateThermoPhaseKernel<<<gridInner, block>>>(d_temperature, d_water_vapor, d_condensed_water, d_width, d_height, d_depth, params);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();



    // Now copy back the data to the CPU
    CUDA_CHECK(cudaMemcpy(Vx, d_Vx, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vy, d_Vy, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vz, d_Vz, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vx_prev, d_Vx_prev, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vy_prev, d_Vy_prev, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vz_prev, d_Vz_prev, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(temperature, d_temperature, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(temp_prev, d_temperature_prev, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(water_vapor, d_water_vapor, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qv_prev, d_qv_prev, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(condensed_water, d_condensed_water, bufferSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qc_prev, d_qc_prev, bufferSize, cudaMemcpyDeviceToHost));
}
