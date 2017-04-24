#ifndef CLOTH_H 
#define CLOTH_H 

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

//Simulation Constants
#define TIME_STEP 0.002 //in seconds
#define NUM_CONSTRAINT_ITERS 1
#define DIFF_CRITICAL 0.2

//Cloth Constants
#define PARTICLE_MASS 1 //in kg
#define STIFFNESS 0.1 //in N/m
#define DAMPING_COEFF 0.005 //in kg/s

//Rendering Constants
#define MIN_BOUND (-2.0f)
#define MAX_BOUND (2.0f)
#define BOUND_LENGTH ((MAX_BOUND) - (MIN_BOUND))

#define LEFT 0
#define RIGHT 1
#define DEBUG
#ifdef DEBUG
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond)
#endif

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
    vector3D color;
    vector3D force;
    vector3D vel;
    bool fixed;
} particle;

typedef struct spring
{
    int left;
    int right;
    float rest_length;
    float k;
    float damping;
    spring_type_t spring_type;
} spring;

class Cloth
{
    private:
        int num_particles_width;
        int num_particles_height;
        int num_springs;
        particle *particles;
        spring *springs;

        void apply_spring_forces();
        void apply_wind_forces();
        void update_positions();
        void apply_forces();
        void satisfy_constraints();
        void render_particles(float rotate_x, float rotate_y, 
                              float translate_z);
        void render_springs(float rotate_x, float rotate_y, float translate_z);
        void make_diagonal_link(int i, int j, int &spring_cnt, int dir, 
                                float len);
        void make_structural_link(int i, int j, int target, int &spring_cnt, 
                                  float len, spring_type_t type);
        void reset_fixed_particles();
        void apply_damping_forces();

    public:
        Cloth(int n = 2);
        Cloth(int w, int h);
        ~Cloth();

        void init();
        void simulate_timestep();
        void render(float rotate_x, float rotate_y, float translate_z);
        inline int get_num_particles();
        inline int get_num_springs();
};


#endif /* CLOTH_H */
