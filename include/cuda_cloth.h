#ifndef CUDA_CLOTH_H 
#define CUDA_CLOTH_H 

#include <string>
#include <stdexcept>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include "cycleTimer.h"
#include "float3.h"

//Simulation Constants
#define TIME_STEP 0.00314 //in seconds
#define NUM_CONSTRAINT_ITERS 300
#define STRETCH_CRITICAL 1.1

//Cloth Constants
#define PARTICLE_MASS 0.01 //in kg

//Spring Constants
#define K_STRUCT 100.0 //in N/m
#define K_SHEAR 100.0
#define K_FLEXION 100.0

#define DAMPING_STRUCT 0.0175
#define DAMPING_SHEAR 0.0175
#define DAMPING_FLEXION 0.0175

//Wind Constants
#define WIND_X 0.0
#define WIND_Y 0.0
#define WIND_Z 1.5

//Rendering Constants
#define MIN_BOUND (-1.0f)
#define MAX_BOUND (1.0f)
#define BOUND_LENGTH ((MAX_BOUND) - (MIN_BOUND))

#define POW_2(base) powf(base, 2.0)

#define LEFT 0
#define RIGHT 1
#define DEBUG
#ifdef DEBUG
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond)
#endif

//cuda constants 
#define TPB_X 32
#define TPB_Y 32
#define TPB (TPB_X * TPB_Y) 

//error checking 
#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define UP_DIV(a, b) (((a) + (b) - 1) / b)

typedef enum
{
    STRUCTURAL,
    SHEAR,
    FLEXION
} spring_type_t;

class cuda_cloth
{
    private:
        int num_particles_width;
        int num_particles_height;
        int num_particles;

        float3 *dev_prev_pos_array;
        float3 *dev_force_array;
        float3 *host_pos_array;

        void update_positions(float3 *dptr);
        void apply_forces(float3 *dptr, float3 *nptr);
        void satisfy_constraints(float3 *dptr);
        void reset_fixed_particles();

    public:
        cuda_cloth(int n = 2);
        cuda_cloth(int w, int h);
        ~cuda_cloth();

        void init();
        void simulate_timestep();
        void render(float rotate_x, float rotate_y, float translate_z);
        inline int get_num_particles();
        inline int get_num_springs();
};


#endif /* CUDA_CLOTH_H */
