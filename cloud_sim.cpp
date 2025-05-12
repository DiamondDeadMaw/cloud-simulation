#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/common.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <functional>
#include <limits>
#include <string>
#include <stdexcept>
#include <chrono>
#include <random>
#include <numeric>



std::random_device rd;
std::mt19937 generator(rd());
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
float totalSimTime = 0.0f;

const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 768;

// const int GRID_SIZE_X = 32;
// const int GRID_SIZE_Y = 16;
// const int GRID_SIZE_Z = 24; 

const int GRID_SIZE_X = 128;
const int GRID_SIZE_Y = 64;
const int GRID_SIZE_Z = 64; 

// World Parameters (what the simulation will see, in meters)
const float worldWidth = 8.0f;
const float worldHeight = worldWidth * ((float)GRID_SIZE_Y / GRID_SIZE_X);
const float worldDepth = worldWidth * ((float)GRID_SIZE_Z / GRID_SIZE_X);
const glm::vec3 volumeDims = glm::vec3(worldWidth, worldHeight, worldDepth);
const glm::vec3 volumeCenter = glm::vec3(0.0f);
// Make the grid uniform
const float CELL_DX = worldWidth / GRID_SIZE_X;
const float CELL_DY = worldHeight / GRID_SIZE_Y;
const float CELL_DZ = worldDepth / GRID_SIZE_Z;


// Simulation parameters

const float DT = 0.05f;
// Diffusion rate for qc (cloud water)
const float DENSITY_DIFFUSION = 0.000005f;
// Diffusion rate for qv (water vapor)
const float VAPOR_DIFFUSION = 0.00002f;
const float TEMP_DIFFUSION = 0.00002f;
const float VISCOSITY = 0.000001f;
const int SOLVER_ITERATIONS = 6;

// Mouse params ====
const float MOUSE_FORCE = 100.0f;
// Amount of qc added by mouse
const float MOUSE_DENSITY_ADD = 0.5f;
// Amount of qv
const float MOUSE_VAPOR_ADD = 2.0f;
const float MOUSE_TEMP_ADD = 5.0f;

// For some noise
const float BACKGROUND_WIND_X = 0.01f;

// Dissipation rate of visible cloud (qc)
const float QC_DISSIPATION = 0.9995f;
// Vapor doesnt dissipate on its own
const float QV_DISSIPATION = 1.0f;
// Temperature doesnt dissipate on its own
const float TEMP_DISSIPATION = 1.0f;


// Cloud physics params ===
// How much temp difference will affect buoyancy
const float BUOYANCY_ALPHA = 0.08f;
// Influence of cloud water (qc) on buoyancy (negative)
const float BUOYANCY_BETA = 0.12f;
// (kg/m^3)
const float BASE_AIR_DENSITY_FOR_BUOYANCY = 1.225f;
// m/s^2
const float GRAVITY_ACCEL = 9.81f;

// Environment params ===
// In kelvin
const float AMBIENT_TEMP_SEA_LEVEL_K = 288.15f;
// kelvin / meter
const float AMBIENT_LAPSE_RATE_K_PER_M = 0.0065f;
// Humidity (vapor) at sea level in kg/kg
const float AMBIENT_VAPOR_SEA_LEVEL = 0.008f;
// Decrease of humidity with height (kg/kg per meter)
const float AMBIENT_VAPOR_DECREASE_RATE = 0.000002f;


// Phase Change Parameters
// Supersaturation to qc conversion rate
const float CONDENSATION_RATE = 0.1f;
// Evaporation rate of qc in subsaturated conditions
const float EVAPORATION_RATE = 0.01f;
// Latent heat of vaporization (J/kg)
const float LATENT_HEAT_VAPORIZATION = 2.501e6f;
// Specific heat of air at constant pressure (J/kg/K)
const float SPECIFIC_HEAT_AIR_PRESSURE = 1005.0f;
// Gas constant for dry air (J/kg/K)
const float GAS_CONSTANT_DRY_AIR = 287.058f;
// Gas constant for water vapor (J/kg/K)
const float GAS_CONSTANT_WATER_VAPOR = 461.5f;
// Ratio C_p / R_d, useful for potential temperature calculations
const float CP_OVER_RD = SPECIFIC_HEAT_AIR_PRESSURE / GAS_CONSTANT_DRY_AIR;
// Ratio R_d / R_v (approx. 0.622), used for vapor pressure calculations
const float EPSILON_RD_OVER_RV = GAS_CONSTANT_DRY_AIR / GAS_CONSTANT_WATER_VAPOR;
const float MOUSE_PUSH_STRENGTH = 25.0f;

// To store final data (densities)
std::vector<float> qcVolumeData(GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z);
std::vector<float> rainVolumeData(GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z);
// Stores texture data to finally render
std::vector<float> frameBuffer(SCR_HEIGHT * SCR_WIDTH * 3);
std::vector<float> hostDetailNoiseVolumeData(GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z * 8); 

GLFWwindow* window = nullptr;
bool mouse_dragging = false;
double last_mouse_x = 0.0, last_mouse_y = 0.0;
double current_mouse_x = 0.0, current_mouse_y = 0.0;
// Mouse will only add density to the singular z slice
int mouse_interaction_slice_z = GRID_SIZE_Z / 2;

glm::vec3 cameraPos   = glm::vec3(0.0f, 0.5f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw   = -90.0f;
float pitch = 0.0f;
float cameraMoveSpeed = 1.5f;
float cameraLookSpeed = 60.0f;
const float FOV = 45.0f;

const glm::vec3 WORLD_UP_AXIS = glm::vec3(0.0f, 1.0f, 0.0f);
const float NEAR_PLANE = 0.1f;
const float FAR_PLANE = 100.0f;

// Environment params
glm::vec3 sunDirection = glm::normalize(glm::vec3(0.8f, 0.6f, -8.0f));
glm::vec3 sunColor = glm::vec3(0.976f, 1.0f, 0.5f);
const float sunExponent = 128.0f;

glm::vec3 zenithColor = glm::vec3(0.1f, 0.2f, 0.4f);
glm::vec3 horizonColor = glm::vec3(0.4f, 0.6f, 0.8f);

// Shaders
const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
out vec2 TexCoords;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoords = aTexCoords;
}

)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoords);
}
)glsl";

// What the ray tracer will draw to
float outputQuadVertices[] = {
    // positions   // texture coords
    -1.0f,  1.0f,  0.0f, 1.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f, 0.0f,

    -1.0f,  1.0f,  0.0f, 1.0f,
     1.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f, 1.0f
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

// Controls rendering
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

    // Light properties
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

// Control physics simulation
struct CudaSimulationParams {
    float DT = 0.05f;
    float DENSITY_DIFFUSION = 0.000005f * 0.5f;
    float VAPOR_DIFFUSION = 0.00002f * 0.5f;
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


extern "C" {
    void initCuda(int imageWidth, int imageHeight, int grid_x, int grid_y, int grid_z);
    void cleanUpCuda();
    void computeFrame(float* hostFrameBuffer, float* hostCloudDensityField, float* hostRainDensityField, float* hostDetailNoiseField,const CudaParams& params);
    void initGpuSimulation(int grid_x, int grid_y, int grid_z);
    void cleanupSimulationCuda();
    void updateSimGPU(float* Vx, float* Vy, float* Vz, float* Vx_prev, float* Vy_prev, float* Vz_prev,
        float* temperature, float* water_vapor, float* condensed_water, float* temp_prev, float* qv_prev, float* qc_prev,  CudaSimulationParams& params
    );
}


// Kelvin to pascals
float calculate_saturation_vapor_pressure(float T_kelvin) {
    if (T_kelvin <= 0) return 0.0f;
    float T_celsius = T_kelvin - 273.15f;
    return 611.2f * exp((17.67f * T_celsius) / (T_celsius + 243.5f));
}

// K,Pa -> kg/kg
float calculate_saturation_mixing_ratio(float T_kelvin, float p_pascal) {
    if (p_pascal <= 0) return 0.0f;
    float es = calculate_saturation_vapor_pressure(T_kelvin);
    float denom = p_pascal - es;
    if (denom <= 1e-6) return 1.0f;
    return std::max(0.0f, (EPSILON_RD_OVER_RV * es) / denom);
}

float get_ambient_temperature(int j_index) {
    float height_m = (j_index - 0.5f) * CELL_DY;
     height_m = std::max(0.0f, height_m);
    return AMBIENT_TEMP_SEA_LEVEL_K - AMBIENT_LAPSE_RATE_K_PER_M * height_m;
}

float get_ambient_vapor(int j_index) {
     float height_m = (j_index - 0.5f) * CELL_DY;
     height_m = std::max(0.0f, height_m);
     return std::max(0.0f, AMBIENT_VAPOR_SEA_LEVEL - AMBIENT_VAPOR_DECREASE_RATE * height_m);
}

float estimate_pressure(int j_index) {
     float height_m = (j_index - 0.5f) * CELL_DY;
     height_m = std::max(0.0f, height_m);
     const float scale_height = 8000.0f;
     return 101325.0f * exp(-height_m / scale_height);
}


class FluidGrid3D {
public:
    // X,Y,Z
    int width, height, depth;
    int size;
    float dt;
    float visc;
    float qc_diff;
    float qv_diff;
    float temp_diff;


    std::vector<float> Vx, Vy, Vz;
    std::vector<float> Vx_prev, Vy_prev, Vz_prev;

    std::vector<float> temperature, temperature_prev;
    // Mixing ratio
    std::vector<float> water_vapor, qv_prev;
    // Mixing ratio
    std::vector<float> condensed_water, qc_prev;
    std::vector<float> rain_density_field;

    FluidGrid3D(int w, int h, int d, float viscosity, float qc_diffusion, float qv_diffusion, float temp_diffusion, float dt)
        : width(w + 2), height(h + 2), depth(d + 2),
          size(this->width * this->height * this->depth),
          dt(dt), visc(viscosity), qc_diff(qc_diffusion), qv_diff(qv_diffusion), temp_diff(temp_diffusion),
          Vx(size, 0.0f), Vy(size, 0.0f), Vz(size, 0.0f),
          Vx_prev(size, 0.0f), Vy_prev(size, 0.0f), Vz_prev(size, 0.0f),
          temperature(size), temperature_prev(size),
          water_vapor(size, 0.0f), qv_prev(size, 0.0f),
          condensed_water(size, 0.0f), qc_prev(size, 0.0f)
    {
         for(int j=0; j < this->height; ++j) {
              float ambient_T = get_ambient_temperature(j);
              for(int k=0; k < this->depth; ++k) {
                   for(int i=0; i < this->width; ++i) {
                        temperature[IX(i,j,k)] = ambient_T;
                   }
              }
         }
         for(int j=0; j < this->height; ++j) {
              float ambient_qv = get_ambient_vapor(j);
              for(int k=0; k < this->depth; ++k) {
                   for(int i=0; i < this->width; ++i) {
                        water_vapor[IX(i,j,k)] = ambient_qv;
                   }
              }
         }
         rain_density_field.resize(GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z, 0.0f);
         initGpuSimulation(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
    }
    ~FluidGrid3D() = default;

    // Helper to get flattened index based on grid pos
    int IX(int x, int y, int z) const {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        z = std::max(0, std::min(z, depth - 1));
        return x + y * width + z * width * height;
    }

    int getRainFlatIndex(int x, int y, int z) const {
    if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y || z < 0 || z >= GRID_SIZE_Z) {
        return -1;
    }
    return x + y * GRID_SIZE_X + z * GRID_SIZE_X * GRID_SIZE_Y;
    }


    void addCondensedWater(int x, int y, int z, float amount) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1 && z >= 1 && z < depth - 1) {
            condensed_water[IX(x, y, z)] = std::max(0.0f, condensed_water[IX(x, y, z)] + amount);
        }
    }

    void addWaterVapor(int x, int y, int z, float amount) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1 && z >= 1 && z < depth - 1) {
            water_vapor[IX(x, y, z)] = std::max(0.0f, water_vapor[IX(x, y, z)] + amount);
        }
    }

     void addTemperature(int x, int y, int z, float amount) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1 && z >= 1 && z < depth - 1) {
            temperature[IX(x, y, z)] += amount;
        }
    }

    void addVelocity(int x, int y, int z, float amountX, float amountY, float amountZ) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1 && z >= 1 && z < depth - 1) {
            int index = IX(x, y, z);
            Vx[index] += amountX;
            Vy[index] += amountY;
            Vz[index] += amountZ;
        }
    }

void FluidGrid3D::updateRainFieldCPU(float p_dt,
                                   float qc_threshold_for_rain,
                                   // The density value to assign to a new rain particle
                                   float rain_packet_density_value,
                                   float rain_fall_speed_m_per_s,
                                   // Fractional evaporation per second
                                   float rain_evaporation_rate_per_sec,
                                   float rain_spawn_probability_per_cell_per_sec) {

    // Start with current rain for evaporation
    std::vector<float> next_rain_state = rain_density_field;

    for (int k_sim = 0; k_sim < GRID_SIZE_Z; ++k_sim) {
        for (int j_sim = 0; j_sim < GRID_SIZE_Y; ++j_sim) {
            for (int i_sim = 0; i_sim < GRID_SIZE_X; ++i_sim) {
                int current_flat_idx = getRainFlatIndex(i_sim, j_sim, k_sim);
                if (current_flat_idx == -1) continue;

                // Evaporation (applied to whatever rain was in the cell from last step)
                float current_rain_in_cell = rain_density_field[current_flat_idx];
                float evaporated_rain = current_rain_in_cell * rain_evaporation_rate_per_sec * p_dt;
                next_rain_state[current_flat_idx] = std::max(0.0f, current_rain_in_cell - evaporated_rain);

                // Stochastic Generation (adds to the potentially evaporated value)
                int padded_grid_idx = IX(i_sim + 1, j_sim + 1, k_sim + 1);
                float qc = condensed_water[padded_grid_idx];

                if (qc > qc_threshold_for_rain) {
                    float spawn_chance_this_dt = rain_spawn_probability_per_cell_per_sec * p_dt;
                    if (distribution(generator) < spawn_chance_this_dt) {
                        next_rain_state[current_flat_idx] = rain_packet_density_value;
                    }
                }
            }
        }
    }

    std::fill(rain_density_field.begin(), rain_density_field.end(), 0.0f);

    // Calculate how many full cells to shift down.
    float fall_dist_m = rain_fall_speed_m_per_s * p_dt;
    int cells_to_shift_down = static_cast<int>(std::floor(fall_dist_m / CELL_DY));
    
    if (cells_to_shift_down < 0) cells_to_shift_down = 0;

    for (int k_sim = 0; k_sim < GRID_SIZE_Z; ++k_sim) {
        for (int i_sim = 0; i_sim < GRID_SIZE_X; ++i_sim) {
            for (int j_source_sim = GRID_SIZE_Y - 1; j_source_sim >= 0; --j_source_sim) {
                int src_flat_idx = getRainFlatIndex(i_sim, j_source_sim, k_sim);
                if (src_flat_idx == -1 || next_rain_state[src_flat_idx] <= 0.0f) {
                    continue;
                }

                int j_dest_sim = j_source_sim - cells_to_shift_down;

                if (j_dest_sim >= 0 && j_dest_sim < GRID_SIZE_Y) {
                    int dest_flat_idx = getRainFlatIndex(i_sim, j_dest_sim, k_sim);
                    if (dest_flat_idx != -1) {
                        rain_density_field[dest_flat_idx] += next_rain_state[src_flat_idx];
                        rain_density_field[dest_flat_idx] = std::min(rain_density_field[dest_flat_idx], rain_packet_density_value * 1.5f);
                    }
                }
            }
        }
    }
}

void step() {
         // apply slight disappation
         for(int i = 0; i < size; ++i) {
             condensed_water[i] *= QC_DISSIPATION;
         }

        CudaSimulationParams s_params;
        s_params.CELL_DY = CELL_DY;
        updateSimGPU(Vx.data(), Vy.data(), Vz.data(), Vx_prev.data(), Vy_prev.data(), Vz_prev.data(),temperature.data(),water_vapor.data(), 
                    condensed_water.data(), temperature_prev.data(), qv_prev.data(), qc_prev.data(), s_params);

    }
};


class PerlinNoise {
public:
    PerlinNoise(unsigned int seed = 0) {
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        p.resize(256);
        std::iota(p.begin(), p.end(), 0);

        std::shuffle(p.begin(), p.end(), std::mt19937(seed));

        // Duplicate the permutation vector to avoid buffer-overflow issues when interpolating coordinates by joining p to itself
        p.insert(p.end(), p.begin(), p.end());
    }

    // Generate 3D Perlin noise, output is approximately in [-1, 1]
    double noise(double x, double y, double z) const {
        // Find the unit cube that contains the point
        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;

        // Find relative x, y, z of point in cube
        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);

        // Compute fade curves for each of x, y, z
        double u = fade(x);
        double v = fade(y);
        double w = fade(z);

        // Hash coordinates of the 8 cube corners
        int A = p[X] + Y;
        int AA = p[A] + Z;
        int AB = p[A + 1] + Z;
        int B = p[X + 1] + Y;
        int BA = p[B] + Z;
        int BB = p[B + 1] + Z;

        // Add blended results from 8 corners of the cube
        double res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                         grad(p[BA], x - 1, y, z)),
                                 lerp(u, grad(p[AB], x, y - 1, z),
                                         grad(p[BB], x - 1, y - 1, z))),
                         lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                                         grad(p[BA + 1], x - 1, y, z - 1)),
                                 lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                                         grad(p[BB + 1], x - 1, y - 1, z - 1))));
        return res;
    }

private:
    // Permutation table
    std::vector<int> p;
    // Ken Perlins improved fade function: 6t^5 - 15t^4 + 10t^3
    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    // Calculates the dot product of a randomly chosen gradient vector and the vector from the input coordinate to the 8 surrounding points in its unit cube
    static double grad(int hash, double x, double y, double z) {
        // Convert low 4 bits of hash code into 12 gradient directions
        int h = hash & 15; 
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
};

// octaves: number of noise layers
// persistence: how much each octave contributes (amplitude multiplier)
// lacunarity: how much frequency increases for each octave
double fbm_noise(double x, double y, double z, int octaves, double persistence, double lacunarity, const PerlinNoise& pn_instance) {
    double total = 0.0;
    double frequency = 1.0;
    double amplitude = 1.0;
    double maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        total += pn_instance.noise(x * frequency, y * frequency, z * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if (maxValue == 0.0) return 0.0;
    // Normalize total to [0,1]
    return total / maxValue;
}

PerlinNoise global_perlin_noise_generator(512412);
PerlinNoise detail_noise_perlin_generator(44123);

void generateDetailNoiseTextureCPU(
    std::vector<float>& noiseDataOutput,
    const PerlinNoise& pn_gen_instance,
    float p_detailNoiseScale,
    const glm::vec3& p_volumeMin,
    int gridX, int gridY, int gridZ,
    float cellDX, float cellDY, float cellDZ)
{
    std::cout << "Generating 3D detail noise texture on CPU" << std::endl;
    noiseDataOutput.resize(gridX * gridY * gridZ);

    const int octaves = 4;
    const float persistence = 0.5f;
    const float lacunarity = 2.0f;

    for (int k = 0; k < gridZ; ++k) {
        for (int j = 0; j < gridY; ++j) {
            for (int i = 0; i < gridX; ++i) {
                // Calculate center of the current voxel in world coordinates
                double world_x = p_volumeMin.x + (static_cast<double>(i) + 0.5) * cellDX;
                double world_y = p_volumeMin.y + (static_cast<double>(j) + 0.5) * cellDY;
                double world_z = p_volumeMin.z + (static_cast<double>(k) + 0.5) * cellDZ;

                // Apply detailNoiseScale to the coordinates for FBM sampling
                double scaled_wx = world_x * p_detailNoiseScale;
                double scaled_wy = world_y * p_detailNoiseScale;
                double scaled_wz = world_z * p_detailNoiseScale;

                float noise_val_neg1_to_1 = static_cast<float>(fbm_noise(scaled_wx, scaled_wy, scaled_wz, octaves, persistence, lacunarity, pn_gen_instance));

                float noise_val_0_to_1 = (noise_val_neg1_to_1 * 0.5f) + 0.5f;

                int flat_idx = i + j * gridX + k * gridX * gridY;
                noiseDataOutput[flat_idx] = noise_val_0_to_1;
            }
        }
    }
    std::cout << "CPU Detail noise texture generation complete" << std::endl;
}

unsigned int shaderProgram;
unsigned int quadVBO, quadVAO;
unsigned int cpuTexture;
float cloudOpacityMultiplierUniform = 150.0f;

FluidGrid3D fluidGrid(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z, VISCOSITY, DENSITY_DIFFUSION, VAPOR_DIFFUSION, TEMP_DIFFUSION, DT);

void framebuffer_size_callback(GLFWwindow* win, int width, int height) { 
    glViewport(0, 0, width, height); 
}

unsigned int compileShader(GLenum type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    int success; 
    char infoLog[512]; 
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) { 
        glGetShaderInfoLog(shader, 512, NULL, infoLog); 
        std::string shaderType = (type == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT"); 
        std::cerr << "ERROR::SHADER::" << shaderType << "::COMPILATION_FAILED\n" << infoLog << std::endl; 
        glDeleteShader(shader); return 0; 
    }
    return shader;
}
unsigned int createShaderProgram(const char* vsSource, const char* fsSource) {
     unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vsSource);
     unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSource);
     if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
     }
     unsigned int program = glCreateProgram(); 
     glAttachShader(program, vertexShader); 
     glAttachShader(program, fragmentShader); 
     glLinkProgram(program);
     int success; 
     char infoLog[512]; 
     glGetProgramiv(program, GL_LINK_STATUS, &success);
     if (!success) { 
        glGetProgramInfoLog(program, 512, NULL, infoLog); 
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl; 
    }
     glDeleteShader(vertexShader); 
     glDeleteShader(fragmentShader);
     return program;
}

bool intersectRayAABB(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax, float& t_min_intersect) {
    glm::vec3 invDir;
    // Handle cases where ray direction components are zero to avoid division by zero / NaNs
    invDir.x = (ray.direction.x == 0.0f) ? std::numeric_limits<float>::infinity() * (ray.direction.x >= 0 ? 1 : -1) : 1.0f / ray.direction.x;
    invDir.y = (ray.direction.y == 0.0f) ? std::numeric_limits<float>::infinity() * (ray.direction.y >= 0 ? 1 : -1) : 1.0f / ray.direction.y;
    invDir.z = (ray.direction.z == 0.0f) ? std::numeric_limits<float>::infinity() * (ray.direction.z >= 0 ? 1 : -1) : 1.0f / ray.direction.z;

    glm::vec3 t0s = (aabbMin - ray.origin) * invDir;
    glm::vec3 t1s = (aabbMax - ray.origin) * invDir;

    glm::vec3 tmin_s = glm::min(t0s, t1s);
    glm::vec3 tmax_s = glm::max(t0s, t1s);

    float t_enter = glm::max(glm::max(tmin_s.x, tmin_s.y), tmin_s.z);
    float t_exit = glm::min(glm::min(tmax_s.x, tmax_s.y), tmax_s.z);

    if (t_enter < t_exit && t_exit >= 0.0f) {
        t_min_intersect = std::max(0.0f, t_enter);
        return true;
    }
    return false;
}


void mouse_button_callback(GLFWwindow* win, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) { 
            mouse_dragging = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
            current_mouse_x = last_mouse_x;
            current_mouse_y = last_mouse_y;

        }
        else if (action == GLFW_RELEASE) { 
            mouse_dragging = false; 
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        float ndc_x = (2.0f * static_cast<float>(current_mouse_x)) / SCR_WIDTH - 1.0f;
        float ndc_y = 1.0f - (2.0f * static_cast<float>(current_mouse_y)) / SCR_HEIGHT;

        glm::mat4 projectionMatrix = glm::perspective(glm::radians(FOV), (float)SCR_WIDTH / (float)SCR_HEIGHT, NEAR_PLANE, FAR_PLANE);
        glm::mat4 invProjectionMatrix = glm::inverse(projectionMatrix);
        
        glm::vec4 ray_clip = glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
        glm::vec4 ray_eye = invProjectionMatrix * ray_clip;
        ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

        glm::mat4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp); 
        glm::mat4 invViewMatrix = glm::inverse(viewMatrix);
        glm::vec3 ray_world_dir = glm::normalize(glm::vec3(invViewMatrix * ray_eye));

        Ray clickRay;
        clickRay.origin = cameraPos;
        clickRay.direction = ray_world_dir;

        glm::vec3 gridVolumeMin = volumeCenter - volumeDims / 2.0f;
        glm::vec3 gridVolumeMax = volumeCenter + volumeDims / 2.0f;
        float t_intersect;

        if (intersectRayAABB(clickRay, gridVolumeMin, gridVolumeMax, t_intersect)) {
            glm::vec3 intersectionPointWorld = clickRay.origin + clickRay.direction * t_intersect;

            glm::vec3 pointInGridSystem = intersectionPointWorld - gridVolumeMin;
            
            int hit_center_x = static_cast<int>(pointInGridSystem.x / CELL_DX);
            int hit_center_y = static_cast<int>(pointInGridSystem.y / CELL_DY);
            int hit_center_z = static_cast<int>(pointInGridSystem.z / CELL_DZ);

            hit_center_x = std::max(0, std::min(hit_center_x, GRID_SIZE_X - 1));
            hit_center_y = std::max(0, std::min(hit_center_y, GRID_SIZE_Y - 1));
            hit_center_z = std::max(0, std::min(hit_center_z, GRID_SIZE_Z - 1));

            const int effect_radius = 1; 
            for (int dz = -effect_radius; dz <= effect_radius; ++dz) {
                for (int dy = -effect_radius; dy <= effect_radius; ++dy) {
                    for (int dx = -effect_radius; dx <= effect_radius; ++dx) {
                        int current_grid_x = hit_center_x + dx;
                        int current_grid_y = hit_center_y + dy;
                        int current_grid_z = hit_center_z + dz;

                        // Convert 0-indexed grid coords to 1-indexed sim coords for adding velocity
                        int sim_x = current_grid_x + 1;
                        int sim_y = current_grid_y + 1;
                        int sim_z = current_grid_z + 1;

                        // Check if the cell is within the valid simulation domain (excluding padding)
                        if (sim_x >= 1 && sim_x <= GRID_SIZE_X &&
                            sim_y >= 1 && sim_y <= GRID_SIZE_Y &&
                            sim_z >= 1 && sim_z <= GRID_SIZE_Z) {
                            
                            glm::vec3 push_velocity = clickRay.direction * MOUSE_PUSH_STRENGTH;
                            fluidGrid.addVelocity(sim_x, sim_y, sim_z, push_velocity.x, push_velocity.y, push_velocity.z);
                        }
                    }
                }
            }

        } 
    }
}

void cursor_position_callback(GLFWwindow* win, double xpos, double ypos) {
    if (mouse_dragging) {
    current_mouse_x = xpos;
    current_mouse_y = ypos;
} else {
    current_mouse_x = xpos;
    current_mouse_y = ypos;
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}
}
bool isRaining = false;
bool g_rKeyPressedLastFrame = false;

void processInput(GLFWwindow *win) {
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(win, true);

    float currentFrameMoveSpeed = cameraMoveSpeed * DT;
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += currentFrameMoveSpeed * cameraFront;
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= currentFrameMoveSpeed * cameraFront;

    glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
    glm::vec3 localCameraRight = glm::normalize(glm::cross(cameraFront, WORLD_UP_AXIS));
    // Handle singularity if cameraFront is parallel to WORLD_UP_AXIS
    if (glm::abs(glm::dot(cameraFront, WORLD_UP_AXIS)) > 0.999f) {
        glm::vec3 tempWorldUpFallback = glm::vec3(1.0f, 0.0f, 0.0f);
        if (glm::abs(glm::dot(cameraFront, tempWorldUpFallback)) > 0.999f) {
            tempWorldUpFallback = glm::vec3(0.0f,0.0f,1.0f);
        }
        localCameraRight = glm::normalize(glm::cross(cameraFront, tempWorldUpFallback));
    }
    cameraUp = glm::normalize(glm::cross(localCameraRight, cameraFront));

    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= localCameraRight * currentFrameMoveSpeed;
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += localCameraRight * currentFrameMoveSpeed;
    if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
         cameraPos += currentFrameMoveSpeed * WORLD_UP_AXIS;
    if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
         cameraPos -= currentFrameMoveSpeed * WORLD_UP_AXIS;

    float currentFrameLookSpeed = cameraLookSpeed * DT * 0.7f; 
    if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)
        pitch += currentFrameLookSpeed;
    if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS)
        pitch -= currentFrameLookSpeed;
    if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS)
        yaw -= currentFrameLookSpeed;
    if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS)
        yaw += currentFrameLookSpeed;

    // Clamp pitch to avoid flipping
    if(pitch > 89.0f) pitch = 89.0f;
    if(pitch < -89.0f) pitch = -89.0f;

    // Recalculate front vector based on updated yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    if (mouse_dragging) {
        // Normalize mouse coordinates to grid coordinates (0 to GRID_SIZE)- need to add a 1, since grid is 1-indexed
        float gridXf = (current_mouse_x / SCR_WIDTH) * GRID_SIZE_X + 1.0f;
        float gridYf = ((SCR_HEIGHT - current_mouse_y) / SCR_HEIGHT) * GRID_SIZE_Y + 1.0f;
        int gridZ = mouse_interaction_slice_z + 1;
        float dx = (float)(current_mouse_x - last_mouse_x);
        // invert y
        float dy = (float)(last_mouse_y - current_mouse_y);

        int cX = static_cast<int>(gridXf);
        int cY = static_cast<int>(gridYf);

        cX = std::max(1, std::min(cX, GRID_SIZE_X));
        cY = std::max(1, std::min(cY, GRID_SIZE_Y));
        gridZ = std::max(1, std::min(gridZ, GRID_SIZE_Z));

        fluidGrid.addWaterVapor(cX, cY, gridZ, MOUSE_VAPOR_ADD * DT);
        fluidGrid.addTemperature(cX, cY, gridZ, MOUSE_TEMP_ADD * DT);
        fluidGrid.addCondensedWater(cX, cY, gridZ, MOUSE_DENSITY_ADD * DT);
        float forceX = dx * MOUSE_FORCE * DT;
        float forceY = dy * MOUSE_FORCE * DT;
        fluidGrid.addVelocity(cX, cY, gridZ, forceX, forceY, 0.0f);
        // Apply to neightbours
        for (int offX = -1; offX <= 1; ++offX) {
            for (int offY = -1; offY <= 1; ++offY) {
                 if (offX == 0 && offY == 0) continue;
                 float falloff = 0.3f;
                 fluidGrid.addWaterVapor(cX + offX, cY + offY, gridZ, MOUSE_VAPOR_ADD * DT * falloff);
                 fluidGrid.addTemperature(cX + offX, cY + offY, gridZ, MOUSE_TEMP_ADD * DT * falloff);
                 fluidGrid.addCondensedWater(cX + offX, cY + offY, gridZ, MOUSE_DENSITY_ADD * DT * falloff);
                 fluidGrid.addVelocity(cX + offX, cY + offY, gridZ, forceX * falloff, forceY * falloff, 0.0f);
            }
        }
        last_mouse_x = current_mouse_x;
        last_mouse_y = current_mouse_y;
    }

    bool rKeyPressedThisFrame = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
    if (rKeyPressedThisFrame && !g_rKeyPressedLastFrame) {
        isRaining = !isRaining;
        if (isRaining) std::cout << "Rain ON" << std::endl;
        else std::cout << "Rain OFF" << std::endl;
    }
    g_rKeyPressedLastFrame = rKeyPressedThisFrame;


}


bool initOpenGL() {
    // Set up GLFW
    if (!glfwInit()) { 
        std::cerr << "Failed to initialize GLFW" << std::endl; return false; 
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Clouds", NULL, NULL);
    if (!window) { 
        std::cerr << "Failed to create GLFW window" << std::endl; glfwTerminate(); return false; 
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { 
        std::cerr << "Failed to initialize GLAD" << std::endl; return false; 
    }

    // Set up shader ====
    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (shaderProgram == 0) {
        return false;
    }

    // Set up quad buffer for rendering
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(outputQuadVertices), outputQuadVertices, GL_STATIC_DRAW);

    // Position verts
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4* sizeof(float), (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Create the fullscreen texture ====
    glGenTextures(1, &cpuTexture);
    glBindTexture(GL_TEXTURE_2D, cpuTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Dont need for ray tracing
    glDisable(GL_DEPTH_TEST);

    return true;
}


// Helper to generate a pseudo-random offset within a cell using Perlin noise
glm::vec3 getCellFeatureOffset_CPU(int ix, int iy, int iz, const PerlinNoise& pn_instance) {
    // Using different large offsets for each component to get different noise streams
    float offsetX = static_cast<float>((pn_instance.noise(ix * 17.371, iy * 23.119, iz * 31.537) + 1.0) * 0.5);
    float offsetY = static_cast<float>((pn_instance.noise(ix * 19.893 + 100.0, iy * 29.431 + 200.0, iz * 37.719 + 300.0) + 1.0) * 0.5);
    float offsetZ = static_cast<float>((pn_instance.noise(ix * 23.677 - 100.0, iy * 31.193 - 200.0, iz * 41.031 - 300.0) + 1.0) * 0.5);
    return glm::vec3(offsetX, offsetY, offsetZ);
}

// out_F1: will store the distance to the closest point
// out_F2: will store the distance to the second closest point
void worleyNoiseF1F2(double x, double y, double z, double frequency, const PerlinNoise& pn_instance, float& out_F1, float& out_F2) {
    // Scale sample point by frequency to work in "cell space"
    double sx = x * frequency;
    double sy = y * frequency;
    double sz = z * frequency;

    // Get integer cell coordinates of the sample point
    int ix = static_cast<int>(std::floor(sx));
    int iy = static_cast<int>(std::floor(sy));
    int iz = static_cast<int>(std::floor(sz));

    float minDistSq1 = std::numeric_limits<float>::max();
    float minDistSq2 = std::numeric_limits<float>::max();

    // Iterate over a 3x3x3 neighborhood of cells
    for (int oz = -1; oz <= 1; ++oz) {
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                int cell_ix = ix + ox;
                int cell_iy = iy + oy;
                int cell_iz = iz + oz;

                // Get the pseudo-random feature point for this cell
                glm::vec3 feature_offset = getCellFeatureOffset_CPU(cell_ix, cell_iy, cell_iz, pn_instance);

                // Calculate the world space position of the feature point
                glm::vec3 featurePoint(
                    (static_cast<float>(cell_ix) + feature_offset.x) / static_cast<float>(frequency),
                    (static_cast<float>(cell_iy) + feature_offset.y) / static_cast<float>(frequency),
                    (static_cast<float>(cell_iz) + feature_offset.z) / static_cast<float>(frequency)
                );

                // Calculate squared distance from sample point to this feature point
                glm::vec3 d_vec = glm::vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)) - featurePoint;
                float distSq = glm::dot(d_vec, d_vec);

                // Update F1 and F2
                if (distSq < minDistSq1) {
                    minDistSq2 = minDistSq1;
                    minDistSq1 = distSq;
                } else if (distSq < minDistSq2) {
                    minDistSq2 = distSq;
                }
            }
        }
    }

    out_F1 = std::sqrt(minDistSq1);
    out_F2 = std::sqrt(minDistSq2);
}



void createSingleCumulusInflow(
    FluidGrid3D& grid,
    int cx_grid, int cy_grid, int cz_grid,
    float base_radius_cells,
    float height_factor,
    float baseMaxQv, float baseTempIncrease, float initialHorizVelocity,

    float base_shape_freq,
    float billow_worley_freq,
    float detail_fbm_freq,
    int   fbm_octaves,
    float fbm_persistence,
    float primary_billow_strength,
    float detail_strength,
    float domain_warp_strength_factor
) {
    static float overall_min_density_before_final_smooth = 1e6f;
    static float overall_max_density_before_final_smooth = -1e6f;
    static int point_counter_smoothstep = 0;

    int r_cells_x = static_cast<int>(base_radius_cells);
    int r_cells_y = static_cast<int>(base_radius_cells * height_factor);
    int r_cells_z = static_cast<int>(base_radius_cells);

    // Define bounding box for iteration
    int startX = std::max(0, cx_grid - r_cells_x * 2); 
    int endX = std::min(GRID_SIZE_X - 1, cx_grid + r_cells_x * 2);
    // Base of cloud can be near cy_grid
    int startY = std::max(0, cy_grid - r_cells_y);
     // Allow for taller growth due to noise
    int endY = std::min(GRID_SIZE_Y - 1, cy_grid + r_cells_y * 2);
    int startZ = std::max(0, cz_grid - r_cells_z * 2);
    int endZ = std::min(GRID_SIZE_Z - 1, cz_grid + r_cells_z * 2);
    // Cloud base world Y, used for vertical profiling
    float cloud_base_y_world = static_cast<float>(cy_grid) * CELL_DY;

    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = startY; j <= endY; ++j) {
        if ((j - startY) % 4 == 0) {
    std::cout << "Processing Y layer " << (j - startY) << " / " << (endY - startY) << std::endl;
}
        for (int k = startZ; k <= endZ; ++k) {
            for (int i = startX; i <= endX; ++i) {
                // Normalized distances from the core center
                // These are relative to the intended core radii
                float norm_dx = (static_cast<float>(i) - cx_grid) / (r_cells_x + 1e-5f);
                // dy from nominal cloud center/base
                float norm_dy_from_base = (static_cast<float>(j) - cy_grid) / (r_cells_y + 1e-5f);
                float norm_dz = (static_cast<float>(k) - cz_grid) / (r_cells_z + 1e-5f);
                // Center slightly up
                float ellipsoid_norm_dy = (static_cast<float>(j) - (cy_grid + 0.2f * r_cells_y)) / (r_cells_y * 0.8f + 1e-5f);
                float dist_sq_ellipsoid = norm_dx * norm_dx + ellipsoid_norm_dy * ellipsoid_norm_dy + norm_dz * norm_dz;

                float base_ellipsoid_density = glm::smoothstep(1.8f, 0.5f, std::sqrt(dist_sq_ellipsoid));
        
                if (base_ellipsoid_density < 0.0001f) continue;

                // World coordinates for noise sampling
                double world_x = static_cast<double>(i) * CELL_DX;
                double world_y = static_cast<double>(j) * CELL_DY;
                double world_z = static_cast<double>(k) * CELL_DZ;

                // Domain warping- curls and irregularities
                float worley_cell_size_approx = 1.0f / (billow_worley_freq + 1e-5f);
                float warp_strength_world = worley_cell_size_approx * domain_warp_strength_factor;
                glm::vec3 warp_offset(0.0f);
                if (warp_strength_world > 1e-5f) {
                    float warp_fbm_freq = base_shape_freq * 0.7f;
                    warp_offset.x = static_cast<float>(fbm_noise(world_x * warp_fbm_freq + 100.0, world_y * warp_fbm_freq, world_z * warp_fbm_freq, 2, 0.5, 2.0, global_perlin_noise_generator));
                    warp_offset.y = static_cast<float>(fbm_noise(world_x * warp_fbm_freq, world_y * warp_fbm_freq + 200.0, world_z * warp_fbm_freq, 2, 0.5, 2.0, global_perlin_noise_generator));
                    warp_offset.z = static_cast<float>(fbm_noise(world_x * warp_fbm_freq, world_y * warp_fbm_freq, world_z * warp_fbm_freq + 300.0, 2, 0.5, 2.0, global_perlin_noise_generator));
                    warp_offset *= warp_strength_world;
                }
                double warped_wx = world_x + warp_offset.x;
                double warped_wy = world_y + warp_offset.y;
                double warped_wz = world_z + warp_offset.z;

                // Shape modulation- distort the ellipsoid
                double shape_fbm_val_raw = fbm_noise(warped_wx * base_shape_freq,
                                                     warped_wy * base_shape_freq,
                                                     warped_wz * base_shape_freq,
                                                     fbm_octaves, fbm_persistence, 2.0,
                                                     global_perlin_noise_generator);
                // Remap FBM from [-1,1] to [0,1]
                float shape_fbm_val = static_cast<float>((shape_fbm_val_raw + 1.0) * 0.5);
                // Modulate the base ellipsoid density. This noise can "eat away" or "add to" the base
                // Values around 0.5 for shape_fbm_val will keep base_ellipsoid_density. Values < 0.5 will reduce, > 0.5 will increase.
                float density = base_ellipsoid_density * glm::mix(0.3f, 1.7f, shape_fbm_val);

                // Primary billows (F2 - F1)
                float F1, F2;
                worleyNoiseF1F2(warped_wx, warped_wy, warped_wz, billow_worley_freq, global_perlin_noise_generator, F1, F2);
                float worley_f2_f1 = F2 - F1;

                // Normalize F2-F1. Positive values are ridges/edges of cells.
                float normalized_f2_f1 = worley_f2_f1 / (worley_cell_size_approx * 0.4f + 1e-5f);
                normalized_f2_f1 = glm::clamp(normalized_f2_f1, 0.0f, 1.0f);

                // Apply billow strength: make density higher at Worley ridges
                float current_cell_y_world = static_cast<float>(j) * CELL_DY;
                float height_above_base_norm = glm::clamp((current_cell_y_world - cloud_base_y_world) / (r_cells_y * CELL_DY + 1e-5f), 0.0f, 1.5f);

                // Make billows stronger higher up and towards the horizontal edges of the base shape
                float radial_dist_norm = std::sqrt(norm_dx * norm_dx + norm_dz * norm_dz);
                float billow_strength_mod = primary_billow_strength * glm::smoothstep(0.1f, 0.7f, height_above_base_norm) * glm::smoothstep(0.5f, 1.0f, radial_dist_norm);
                // Billows are stronger where base FBM is high
                billow_strength_mod *= glm::smoothstep(0.4f, 0.8f, shape_fbm_val);


                // Additive contribution for billows:
                // density += billow_strength_mod * normalized_f2_f1;
                
                // ALTERNATIVE, DONT USE F2- F1 ===================
                float cellSize = 1.0f / billow_worley_freq; 
                // normalize F1 into [0,1] so 0 at center, 1 at walls
                float billow = glm::clamp(F1 / (cellSize * 0.5f), 0.0f, 1.0f);
                // invert it so you get blobs at cell centers:
                billow = 1.0f - billow;
                

                // Remove dangling spheres
                float envelope = base_ellipsoid_density;
                float envelopeMask = glm::smoothstep(0.02f, 0.6f, envelope);
                density += primary_billow_strength * billow * envelopeMask;

                // MULTIPLICATIVE, WILL REMOVE A LOT OF DETAIL ===========
                // density *= (1.0f + billow_strength_mod * normalized_f2_f1);

                // Vertical density profile
                // This defines the cloud base and how density changes with height.
                float vertical_profile = 1.0f;
                // Cloud base: relatively sharp cutoff
                // Slightly softer base start
                vertical_profile *= glm::smoothstep(0.0f, 0.15f, height_above_base_norm);
                // Encourage mid-growth
                vertical_profile *= glm::smoothstep(0.25f, 0.6f, height_above_base_norm);
                // Taper top
                vertical_profile *= glm::smoothstep(1.4f, 0.8f, height_above_base_norm);

                float baseMask = base_ellipsoid_density * shape_fbm_val * vertical_profile;
                baseMask *= glm::smoothstep(0.05f, 0.15f, baseMask);
        
                density *= vertical_profile;

                // Detail noise- high freq fbm for the surface texture
                if (detail_strength > 0.0f) {
                    double detail_fbm_raw = fbm_noise(warped_wx * detail_fbm_freq,
                                                      warped_wy * detail_fbm_freq,
                                                      warped_wz * detail_fbm_freq,
                                                      2, 0.4, 2.2, global_perlin_noise_generator);
                    float detail_fbm_val = static_cast<float>((detail_fbm_raw + 1.0) * 0.5);
                    // Modulate density slightly for surface roughness
                    density *= glm::mix(1.0f - detail_strength, 1.0f + detail_strength, detail_fbm_val);
                }

                // Final clamping and density cutoff
                density = glm::clamp(density, 0.0f, 2.0f);

                // Track for debugging
                overall_min_density_before_final_smooth = std::min(overall_min_density_before_final_smooth, density);
                overall_max_density_before_final_smooth = std::max(overall_max_density_before_final_smooth, density);
                point_counter_smoothstep++;
                
                // Carving out holes in the surface
                // sample a secondary, tighter Worley
                float F1c, F2c;
                float sec_freq = billow_worley_freq * 2.5f;
                worleyNoiseF1F2(warped_wx, warped_wy, warped_wz,
                                    sec_freq, global_perlin_noise_generator,
                                    F1c, F2c);

                // build a cavity mask- bigger F2–F1 means a ridge, remove
                float sec_cellSize = 1.0f / sec_freq;
                float cav = glm::clamp((F2c - F1c) / (sec_cellSize * 0.9f), 0.0f, 1.0f);

                // only carve when inside the ellipsoid
                float interiorMask = glm::smoothstep(0.3f, 0.7f, base_ellipsoid_density);

                float cavity_strength = 0.3f; 
                density *= (1.0f - cav * cavity_strength * interiorMask);

                // tiny sphere billows==========
                float tinyFreq = billow_worley_freq * 4.0f;
                float F1t, F2t;
                worleyNoiseF1F2(warped_wx, warped_wy, warped_wz,
                                    tinyFreq, global_perlin_noise_generator,
                                    F1t, F2t);

                // normalized distance from cell center
                float tinyCell = 1.0f / tinyFreq;
                float tinyBillow = 1.0f - glm::clamp(F1t / (tinyCell * 0.4f), 0.0f, 1.0f);

                // only along the skin- where final density field is near threshold
                float skinMask = glm::smoothstep(0.2f, 0.8f, density);

                // add tiny fluff
                float tiny_strength = 0.2f;
                density += tiny_strength * tinyBillow * skinMask;

                // Marbling ===
                // Perlin‐FBM inside the core
                double fbm_interior = fbm_noise(warped_wx * detail_fbm_freq * 1.5,
                                                warped_wy * detail_fbm_freq * 1.5,
                                                warped_wz * detail_fbm_freq * 1.5,
                                                3, 0.5, 2.2,
                                                global_perlin_noise_generator);
                float interiorFBM = glm::smoothstep(0.4f, 0.6f, static_cast<float>((fbm_interior +1)*0.5));

                // tint density up/down
                float marbling_strength = 0.15f;
                density *= glm::mix(1.0f - marbling_strength, 1.0f + marbling_strength, interiorFBM * interiorMask);

                // Apply a final smoothstep to create a softer edge from the calculated density
                float final_density_value = glm::smoothstep(0.05f, 0.6f, density);

                if (final_density_value > 0.01f) {
                    float qv_to_add = baseMaxQv * final_density_value;
                    float temp_to_add = baseTempIncrease * final_density_value;

                    grid.addWaterVapor(i + 1, j + 1, k + 1, qv_to_add);
                    grid.addTemperature(i + 1, j + 1, k + 1, temp_to_add);
                    if (std::abs(initialHorizVelocity) > 1e-6f) {
                        // Apply velocity more strongly where density is higher
                        grid.addVelocity(i + 1, j + 1, k + 1, initialHorizVelocity * final_density_value, 0.0f, 0.0f);
                    }
                }
            }
        }
    }
    if (point_counter_smoothstep > 100) {
        printf("Density range BEFORE final smoothstep: Min=%.3f Max=%.3f (from %d points)\n",
               overall_min_density_before_final_smooth, overall_max_density_before_final_smooth, point_counter_smoothstep);
    }
    overall_min_density_before_final_smooth = 1e6f;
    overall_max_density_before_final_smooth = -1e6f;
    point_counter_smoothstep = 0;
}

void manageCloudInflow(FluidGrid3D& grid,float targetIntervalApprox = 60.0f)
{
    static float timeSinceLastPuff = targetIntervalApprox * 0.8f;
    timeSinceLastPuff += DT;

    // Caluclate probability of creating a new cloud
    float intervalLowerBound = targetIntervalApprox * 0.7f;
    float intervalUpperBound = targetIntervalApprox * 1.3f;
    bool addPuffThisFrame = false;
    if (timeSinceLastPuff >= intervalLowerBound) {
        if (timeSinceLastPuff >= intervalUpperBound) {
            addPuffThisFrame = true;
        } else {
            float windowDuration = intervalUpperBound - intervalLowerBound;
            float timeIntoWindow = timeSinceLastPuff - intervalLowerBound;
            float probability = (windowDuration > 1e-5f) ? std::min(1.0f, timeIntoWindow / windowDuration) : 1.0f;
            probability = std::pow(probability, 0.7f);
            if (distribution(generator) < probability) {
                addPuffThisFrame = true;
            }
        }
    }

    if (addPuffThisFrame) {
        timeSinceLastPuff = 0.0f;

        int puff_cx = GRID_SIZE_X / 3 + static_cast<int>(distribution(generator) * (GRID_SIZE_X / 3.0f));
        int puff_cy = GRID_SIZE_Y / 4 + static_cast<int>(distribution(generator) * (GRID_SIZE_Y / 5.0f));
        int puff_cz = GRID_SIZE_Z / 3 + static_cast<int>(distribution(generator) * (GRID_SIZE_Z / 3.0f));

        float puff_radius_cells = (GRID_SIZE_X / 9.0f) + distribution(generator) * (GRID_SIZE_X / 10.0f);
        // Encourage taller structures, but allow some variation
        float puff_height_factor = 1.1f + distribution(generator) * 0.7f;

        float puff_base_qv = 0.011f + distribution(generator) * 0.004f;
        float puff_temp_inc = 1.3f + distribution(generator) * 0.8f;
        float puff_horiz_vel = 0.1f + distribution(generator) * 0.003f;

        float cloud_diameter_world = puff_radius_cells * CELL_DX * 2.0f;

        float num_base_features = 10.5f + distribution(generator) * 1.0f;
        float freq_base_shape = num_base_features / cloud_diameter_world;

        // Billows- keeping moderate, these sit on the base lobes
        float num_billow_features = 15.0f + distribution(generator) * 5.0f;
        float freq_billow_worley = num_billow_features / cloud_diameter_world;

        float freq_detail_fbm = freq_billow_worley * (3.0f + distribution(generator) * 2.0f);

        // Strengths
        float strength_primary_billow = 0.4f + distribution(generator) * 0.3f;
        float strength_detail = 0.08f + distribution(generator) * 0.12f;
        // Increase domain warp strength to get larger scale distortions/lobes
        float strength_domain_warp_factor = 1.8f + distribution(generator) * 0.3f;


        std::cout << "Creating Cumulus at: " << puff_cx << "," << puff_cy << "," << puff_cz
                  << " R_base_cells=" << puff_radius_cells << " H_factor=" << puff_height_factor << std::endl;
        std::cout << "  Freqs: Base=" << freq_base_shape << " Billow=" << freq_billow_worley << " Detail=" << freq_detail_fbm << std::endl;
        std::cout << "  Strengths: Billow=" << strength_primary_billow << " Detail=" << strength_detail << " WarpFactor=" << strength_domain_warp_factor << std::endl;

        createSingleCumulusInflow(
            grid, puff_cx, puff_cy, puff_cz,
            puff_radius_cells, puff_height_factor,
            puff_base_qv, puff_temp_inc, puff_horiz_vel,
            freq_base_shape, freq_billow_worley, freq_detail_fbm,
            5, 0.5f,
            strength_primary_billow, strength_detail,
            strength_domain_warp_factor
        );
    }



}


bool blanketGenerated = false;
void initializeCloudBlanket(FluidGrid3D& grid) {
    if (blanketGenerated) {
        return;
    }
    std::cout << "Initializing cloud blanket" << std::endl;
    // Define how many clouds to spread across the X and Z dimensions
    const int num_clouds_x_dim = 3;
    const int num_clouds_z_dim = 3;

    // Calculate spacing for placing clouds
    float x_spacing = static_cast<float>(GRID_SIZE_X) / num_clouds_x_dim;
    float z_spacing = static_cast<float>(GRID_SIZE_Z) / num_clouds_z_dim;

    // Define the vertical band for the cloud blankets base
    float base_y_level_min = GRID_SIZE_Y * 0.35f;
    float base_y_level_max = GRID_SIZE_Y * 0.55f;

    for (int ix = 0; ix < num_clouds_x_dim; ++ix) {
        for (int iz = 0; iz < num_clouds_z_dim; ++iz) {
            // Calculate base center position for this cloud cell
            float ideal_cx = (ix + 0.5f) * x_spacing;
            float ideal_cz = (iz + 0.5f) * z_spacing;

            // Add jitter to make the placement less grid like
            // Jitter can be up to 30-40% of the spacing
            int puff_cx = static_cast<int>(ideal_cx + (distribution(generator) - 0.5f) * x_spacing * 0.6f);
            int puff_cz = static_cast<int>(ideal_cz + (distribution(generator) - 0.5f) * z_spacing * 0.6f);
            // Randomize base height within the defined band
            int puff_cy = static_cast<int>(base_y_level_min + distribution(generator) * (base_y_level_max - base_y_level_min));

            // Clamp positions to be within grid boundaries
            puff_cx = std::max(0, std::min(puff_cx, GRID_SIZE_X - 1));
            puff_cy = std::max(0, std::min(puff_cy, GRID_SIZE_Y - 1));
            puff_cz = std::max(0, std::min(puff_cz, GRID_SIZE_Z - 1));

            // Radius- aim for clouds large enough to overlap with neighbors
            float min_spacing = std::min(x_spacing, z_spacing);
            float puff_radius_cells = (min_spacing * 0.5f) + distribution(generator) * (min_spacing * 0.4f);
            puff_radius_cells = std::max(puff_radius_cells, GRID_SIZE_X / 15.0f);

            float puff_height_factor = 0.7f + distribution(generator) * 0.6f;

            float puff_base_qv = 0.010f + distribution(generator) * 0.0035f;
            float puff_temp_inc = 0.8f + distribution(generator) * 0.7f;
            float puff_horiz_vel = 0.0f;

            float cloud_diameter_world = puff_radius_cells * CELL_DX * 2.0f;
            if (cloud_diameter_world < 1e-5f) cloud_diameter_world = CELL_DX * 2.0f;

            // Noise frequencies- scaled by cloud size
            // Fewer large features for broader shapes
            float num_base_features = 7.0f + distribution(generator) * 3.0f;
            float freq_base_shape = num_base_features / cloud_diameter_world;

            float num_billow_features = 10.0f + distribution(generator) * 5.0f;
            float freq_billow_worley = num_billow_features / cloud_diameter_world;

            float freq_detail_fbm = freq_billow_worley * (2.0f + distribution(generator) * 2.0f);
            // Noise strengths
            float strength_primary_billow = 0.25f + distribution(generator) * 0.25f;
            float strength_detail = 0.04f + distribution(generator) * 0.08f;
            float strength_domain_warp_factor = 0.6f + distribution(generator) * 0.8f;

            std::cout << "  Placing blanket puff at: (" << puff_cx << ", " << puff_cy << ", " << puff_cz
                      << "), R_cells=" << puff_radius_cells << ", H_factor=" << puff_height_factor << std::endl;

            createSingleCumulusInflow(
                grid, puff_cx, puff_cy, puff_cz,
                puff_radius_cells, puff_height_factor,
                puff_base_qv, puff_temp_inc, puff_horiz_vel,
                freq_base_shape, freq_billow_worley, freq_detail_fbm,
                4, 0.45f, strength_primary_billow, strength_detail,
                strength_domain_warp_factor
            );
        }
    }
    std::cout << "Cloud blanket initialization complete." << std::endl;
    blanketGenerated = true;
}


void updateSimulation() {
    totalSimTime+=DT;
    //manageCloudInflow(fluidGrid);
    initializeCloudBlanket(fluidGrid);
    
    // Apply forces- calculates forces based on the state before the step, and adds velocity
    for (int k = 1; k < fluidGrid.depth - 1; ++k) {
        for (int j = 1; j < fluidGrid.height - 1; ++j) { 
            for (int i = 1; i < fluidGrid.width - 1; ++i) {
                int idx = fluidGrid.IX(i, j, k);
                float T_parcel = fluidGrid.temperature[idx];
                float qv_parcel = fluidGrid.water_vapor[idx];
                float qc_parcel = fluidGrid.condensed_water[idx];

                // Get ambient conditions at this height
                float T_amb = get_ambient_temperature(j);
                float qv_amb = get_ambient_vapor(j);

                // Calculate Virtual Temperatures
                float Tv_parcel = T_parcel * (1.0f + 0.608f * qv_parcel);
                float Tv_amb = T_amb * (1.0f + 0.608f * qv_amb);

                // Buoyancy Force based on virtual temperature difference
                float buoyancy_force = 0.0f;
                 if (Tv_amb > 1e-6) {
                    buoyancy_force = BUOYANCY_ALPHA * GRAVITY_ACCEL * ((Tv_parcel / Tv_amb) - 1.0f) * 3.0f;
                 }

                // Water Weight Force (Downward force due to condensed water qc - negative Y)
                // Force = - mass_water * g / mass_air = - (qc * mass_air) * g / mass_air = - qc * g
                float water_weight_force = -BUOYANCY_BETA * GRAVITY_ACCEL * qc_parcel;
                float total_force_y = buoyancy_force + water_weight_force;
                // fluidGrid.addVelocity(i, j, k,0.0f, total_force_y * DT, 0.0f);
            }
        }
    }

    fluidGrid.step();

if (isRaining) {
    float qc_thresh_for_rain_spawn_cpu = 0.00106f;
    float rain_gen_rate_cpu = 1.0f;
    float rain_fall_speed_cpu = 7.0f;
    float rain_evap_rate_cpu = 0.05f;
    float rain_spawn_prob = 0.7f;

    fluidGrid.updateRainFieldCPU(DT, qc_thresh_for_rain_spawn_cpu, rain_gen_rate_cpu, rain_fall_speed_cpu, rain_evap_rate_cpu, rain_spawn_prob);
} else {
    std::fill(fluidGrid.rain_density_field.begin(), fluidGrid.rain_density_field.end(), 0.0f);
}

    // Copy condensed water to the render texture
    for (int k = 0; k < GRID_SIZE_Z; ++k) { 
        for (int j = 0; j < GRID_SIZE_Y; ++j) { 
            for (int i = 0; i < GRID_SIZE_X; ++i) {
                int grid_idx = fluidGrid.IX(i + 1, j + 1, k + 1);
                int tex_idx = i + j * GRID_SIZE_X + k * GRID_SIZE_X * GRID_SIZE_Y;
                qcVolumeData[tex_idx] = fluidGrid.condensed_water[grid_idx];
                 if (std::isnan(qcVolumeData[tex_idx])) {
                    qcVolumeData[tex_idx] = 0.0f; 
                 }
                 

                 // Rain water copy
                 int flat_idx = fluidGrid.getRainFlatIndex(i, j, k);
                if (flat_idx != -1) {
                    rainVolumeData[flat_idx] = fluidGrid.rain_density_field[flat_idx];
                    if (std::isnan(rainVolumeData[flat_idx])) {
                        rainVolumeData[flat_idx] = 0.0f;
                    }
                }
                 
            }
        }
    }
}

void render() {

    CudaParams params;
    params.cameraPos = cameraPos;
    params.cameraPos = cameraPos;
    params.cameraFront = cameraFront;
    params.cameraRight = glm::normalize(glm::cross(params.cameraFront, WORLD_UP_AXIS));
    if (glm::abs(glm::dot(params.cameraFront, WORLD_UP_AXIS)) > 0.999f) {
    glm::vec3 tempWorldUpFallback = glm::vec3(1.0f, 0.0f, 0.0f);
    if (glm::abs(glm::dot(params.cameraFront, tempWorldUpFallback)) > 0.999f) {
            tempWorldUpFallback = glm::vec3(0.0f,0.0f,1.0f);
        }
        params.cameraRight = glm::normalize(glm::cross(params.cameraFront, tempWorldUpFallback));
    }
    params.cameraUp = glm::normalize(glm::cross(params.cameraRight, params.cameraFront));


    params.volumeMin = volumeCenter - volumeDims / 2.0f;
    params.volumeMax = volumeCenter + volumeDims / 2.0f;
    params.volumeDims = volumeDims;

    params.gridDimX = GRID_SIZE_X;
    params.gridDimY = GRID_SIZE_Y;
    params.gridDimZ = GRID_SIZE_Z;

    params.screenDimX = SCR_WIDTH;
    params.screenDimY = SCR_HEIGHT;

    params.cellSize = glm::vec3(CELL_DX, CELL_DY, CELL_DZ);

    params.sunDirection = sunDirection;
    params.sunColor = sunColor;
    params.zenithColor = zenithColor;
    params.horizonColor = horizonColor;
    params.sunExponent = sunExponent;

    params.isRaining = isRaining;


    params.stepSize = CELL_DX/2.0f;
    params.extinctionScale = 7.0f;
    params.anisotropy = 0.8f;
    params.sunIntensity = 30.0f;
    params.shadowStep = params.stepSize * 4.0f;
    params.shadowMaxDist = glm::length(params.volumeDims);

    params.aspectRatio = float(SCR_WIDTH) / float(SCR_HEIGHT);
    params.scale = tan(glm::radians(FOV) / 2.0f);

    params.noiseSeed = 12345u;

    params.isoThreshold = 0.1f;
    params.surfaceScale = 8.0f;


    // Rain parameters
    params.qcThresholdForRainSpawn = 0.0008f;
    params.rainExtinctionScale = 10.0f;
    params.rainAlbedo = 0.5f;
    params.rainFallSpeed = 7.0f;
    params.totalSimTime = totalSimTime;
    params.detailNoiseScale = 0.1f;
    params.detailWeight = 0.002f;
    params.multiScatterScale = 0.02f;

    computeFrame(frameBuffer.data(), qcVolumeData.data(), rainVolumeData.data(), hostDetailNoiseVolumeData.data(),  params);

    glBindTexture(GL_TEXTURE_2D, cpuTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_FLOAT, frameBuffer.data());


    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cpuTexture);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    

    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        std::cerr << "[GL ERROR after draw] 0x" << std::hex << e << std::dec << "\n";
    }

    glBindVertexArray(0);
}



int main() {
    try {
        if (!initOpenGL()) { return -1; }

        std::cout << "Initializing CUDA" << std::endl;
        initCuda(SCR_WIDTH, SCR_HEIGHT, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);

        double lastTime = glfwGetTime();
        int frameCount = 0;
        glm::vec3 initialLocalRight = glm::normalize(glm::cross(cameraFront, WORLD_UP_AXIS));
        cameraUp = glm::normalize(glm::cross(initialLocalRight, cameraFront));

        float tempDetailNoiseScale = 2.0f;
        glm::vec3 tempVolumeMin = volumeCenter - volumeDims / 2.0f;

        generateDetailNoiseTextureCPU(
            hostDetailNoiseVolumeData,
            detail_noise_perlin_generator,
            tempDetailNoiseScale,
            tempVolumeMin,
            GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z,
            CELL_DX, CELL_DY, CELL_DZ
        );



        while (!glfwWindowShouldClose(window)) {
            double currentTime = glfwGetTime();
            printf("New frame========================\n");
            frameCount++;
            if (currentTime - lastTime >= 1.0) {
                // std::cout << "FPS: " << frameCount << std::endl;
                frameCount = 0;
                lastTime = currentTime;
            }


            processInput(window);
            auto start = std::chrono::high_resolution_clock::now();
            updateSimulation();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start);
            std::cout << "Simulation step took " << duration.count() << " ms\n";

            float min_qc = 1e30f, max_qc = -1e30f, sum_qc = 0.0f;
            int count_positive_qc = 0;
            for (size_t i = 0; i < qcVolumeData.size(); ++i) {
                if (qcVolumeData[i] > 1e-7) {
                    min_qc = std::min(min_qc, qcVolumeData[i]);
                    max_qc = std::max(max_qc, qcVolumeData[i]);
                    sum_qc += qcVolumeData[i];
                    count_positive_qc++;
                }
            }
            float avg_qc = (count_positive_qc > 0) ? sum_qc / count_positive_qc : 0.0f;
            std::cout << "qcVolumeData - Min: " << min_qc << " Max: " << max_qc << " Avg (active): " << avg_qc << " Count: " << count_positive_qc << std::endl;

            start = std::chrono::high_resolution_clock::now();
            render();
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double, std::milli>(end - start);
            std::cout << "Render took " << duration.count() << " ms\n";

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        glfwTerminate();
        return -1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
         glfwTerminate();
         return -1;
    }

    // Cleanup
    cleanUpCuda();
    cleanupSimulationCuda();
    glDeleteVertexArrays(1, &quadVAO); glDeleteBuffers(1, &quadVBO);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &cpuTexture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}