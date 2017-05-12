##Cloth Simulation on CPU, GPU and Ebb

### Build Instructions
```
    mkdir build && cd build
    cmake ..
    make
```

### Dependencies
CMake, OpenGL, GLEW, GLUT, CUDA (Optional)

### Running the Simulation
Press space to advance by 1 timestep

Press r to restart the simulation

### Ebb
Ebb is a DSL for writing Physical Simulations that efficiently ports implementations to the CPU and GPU. Installation instructions for Ebb are [here](https://github.com/gilbo/ebb). To run the Ebb version of the simulation, do
```
    ./vdb & //for visualization purposes (only available for CPU)
    ebb [-g] cloth_sim.t
```
