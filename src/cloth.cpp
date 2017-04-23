#include "cloth.h"

Cloth::Cloth(int n)
{
    num_particles_width = n;
    num_particles_height = n;
    int num_particles = n * n;
    num_springs = 0; //TODO: FIXME
    particles = (particle*)malloc(sizeof(particle) * num_particles);
    springs = (spring*)malloc(sizeof(spring) * num_springs);

    if((particles == NULL) || (springs == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
}

Cloth::Cloth(int w, int h)
{
    num_particles_width = w;
    num_particles_height = h;
    int num_particles = w * h;
    num_springs = 0; //TODO: FIXME
    particles = (particle*)malloc(sizeof(particle) * num_particles);
    springs = (spring*)malloc(sizeof(spring) * num_springs);

    if((particles == NULL) || (springs == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
}

Cloth::~Cloth()
{
    free(particles);
    free(springs);
}

void Cloth::init()
{
    //cloth is a sheet in the xz plane

    //set positions of all particles
    for(int i = 0; i < num_particles_width; i++)
    {
        for(int j = 0; j < num_particles_height; j++)
        {
            float x,z;
            x = (float)j/num_particles_width;
            z = (float)i/num_particles_height;
            x = 2*x - 1.0f;
            z = 2*z - 1.0f;

            particles[i*num_particles_width + j].pos = {x, 0.0f, z};
            particles[i*num_particles_width + j].color = {0.5f, 0.5f, 0.5f};
        }
    }

    //create links for each spring

}

void Cloth::render(float rotate_x, float rotate_y, float translate_z)
{
    //transform the cloth's position in the world space based on 
    //camera parameters
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    render_particles(rotate_x, rotate_y, translate_z);
    render_springs(rotate_x, rotate_y, translate_z);
}

void Cloth::render_particles(float rotate_x, float rotate_y, float translate_z)
{
    int num_particles = get_num_particles();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(particle), &(particles[0].pos.x));
    glColorPointer(3, GL_FLOAT, sizeof(particle), &(particles[0].color.x));
    glDrawArrays(GL_POINTS, 0, num_particles);
    glDisableClientState(GL_VERTEX_ARRAY);

    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBufferSubData(GL_ARRAY_BUFFER, 0, num_particles * sizeof(particle), particles);
    //glVertexPointer(3, GL_FLOAT, sizeof(particle), &particles[0]);

    //glEnableClientState(GL_VERTEX_ARRAY);
    //glColor3f(0.0f, 1.0f, 0.0f);
    //glDrawArrays(GL_POINTS, 0, num_particles);
    //glDisableClientState(GL_VERTEX_ARRAY);
}

inline int Cloth::get_num_particles()
{
    return num_particles_width * num_particles_height;
}

inline int Cloth::get_num_springs()
{
    return 0;
}

void Cloth::render_springs(float rotate_x, float rotate_y, float translate_z)
{
}

void Cloth::apply_forces()
{
    //accumulate force of gravity on all particles
    vector3D gravity(0.0f, -9.8f, 0.0f);

    int num_particles = get_num_particles();
    for(int i = 0; i < num_particles; i++)
    {
        particles[i].force = PARTICLE_MASS * gravity;
    }

    apply_wind_forces();
    apply_spring_forces();
    //TODO: FIXME
}

void Cloth::apply_spring_forces()
{
    //TODO: FIXME
}

void Cloth::apply_wind_forces()
{
    //TODO: FIXME
}


void Cloth::satisfy_constraints()
{
    //TODO: FIXME
}

void Cloth::update_positions()
{
    //perform verlet integration
    int num_particles = get_num_particles();
    for(int i = 0; i < num_particles; i++)
    {
        vector3D temp(particles[i].pos);
        vector3D acc = particles[i].force/PARTICLE_MASS;
        particles[i].pos += (particles[i].pos - particles[i].prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        particles[i].prev_pos = temp;
    }
}

void Cloth::simulate_timestep()
{
    apply_forces();
    update_positions();
    satisfy_constraints();
    //TODO: FIXME
}
