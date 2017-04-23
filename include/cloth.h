#ifndef CLOTH_H 
#define CLOTH_H 

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glx.h>
#endif

#define TIME_STEP 0.002

typedef struct vector3D
{
    float x;
    float y;
    float z;
} vector3D;

typedef enum
{
    STRETCH,
    SHEAR,
    BEND
} spring_type_t;

typedef struct particle
{
    vector3D screen_pos;
    vector3D pos;
    vector3D prev_pos;
    vector3D color;
    float mass;
} particle;

typedef struct spring
{
    int left;
    int right;
    float rest_length;
    float k;
    float damping; //TODO: FIXME
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
        //GLuint vbo;

        void apply_spring_forces();
        void apply_wind_forces();
        void update_positions();
        void apply_forces();
        void satisfy_constraints();
        void render_particles(float rotate_x, float rotate_y, float translate_z);
        void render_springs(float rotate_x, float rotate_y, float translate_z);
        void transform_particle_buffer();

    public:
        Cloth(int n = 2);
        Cloth(int w, int h);
        ~Cloth();

        void init();
        void simulate_timestep();
        void render(float rotate_x, float rotate_y, float translate_z);
        inline unsigned get_num_particles();
        inline unsigned get_num_springs();
};


#endif /* CLOTH_H */
