# A physics based volumetric cloud visualizer


**cloud_sim.exe**: Ellipsoid based generation, single Worley and Perlin passes, simple rendering (only in scattering). Includes rain (press `r`)

**cloud_sim_2.exe**: Ellipsoid based cloud generation with multiple layers, complex rendering (multi scattering)

**cloud_sim_no_ellipse.exe**: Uses spheres for generation — no base shape. Same complex rendering

**cloud_sim_simple.exe**: Renders raw noise generated density shape



To compile: Run build.bat. Ensure you have compute capability 12+, nvcc, and Visual C++ Build Tools (MSVC C++ compiler, cl.exe).

Will need to edit build to point to the correct cuda installation, and MSVC installation paths for your device.

Not tested on linux, should be simpler.

To run exe's: Need to run in the current directory. Requires CUDA compatible GPU.