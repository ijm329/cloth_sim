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
#include "vector3D.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

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

typedef struct GlobalConstants
{
    int num_particles_width;
    int num_particles_height;
    particle *dev_particles;
} GlobalConstants;

typedef enum
{
    STRUCTURAL,
    SHEAR,
    FLEXION
} spring_type_t;

typedef struct particle
{
    vector3D pos;
    vector3D prev_pos;
    vector3D force;
    vector3D normal;
    bool fixed;
} particle;

class CudaCloth
{
    private:
        int num_particles_width;
        int num_particles_height;
        int num_particles;
        particle *particles;
        particle *dev_particles;

        void update_positions();
        void apply_forces();
        void satisfy_constraints();
        void render_springs(float rotate_x, float rotate_y, float translate_z);
        void reset_fixed_particles();
        void reset_normals();
        vector3D get_normal_vec(vector3D p1, vector3D p2, vector3D p3);
        void draw_triangle(particle *p1, particle *p2, particle *p3);
        void draw_square(int curr_idx, int right_idx, int lower_idx, int diag_idx);

    public:
        CudaCloth(int n = 2);
        CudaCloth(int w, int h);
        ~CudaCloth();

        void init();
        void simulate_timestep();
        void render(float rotate_x, float rotate_y, float translate_z);
        inline int get_num_particles();
        inline int get_num_springs();
};


#endif /* CUDA_CLOTH_H */
