# A physics based volumetric cloud visualizer


cloud_sim.exe: ellipsoid based generation, single worley and perlin passes, simple rendering (only in scattering). Includes rain (press r)
cloud_sim_2.exe: Ellipsoid based cloud generation with multiple layers, complex rendering (multi scattering)
cloud_sim_no_ellipse.exe: use spheres for generation- no base shape. Same complex rendering
cloud_sim_simple.exe: Only renders 


To compile: Run build.bat. Ensure you have compute capability 12+, nvcc, and visual c runtimes.
To run exe's: Need to run in the current directory. Requires CUDA compatible GPU.