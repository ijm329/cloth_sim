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
    //set positions of all particles
    for(int i = 0; i < num_particles_width; i++)
    {
        for(int j = 0; j < num_particles_height; j++)
        {
            particles[i*num_particles_width + j].prev_pos = {0.0f, 0.0f, 0.0f};
            particles[i*num_particles_width + j].pos = {j, 0.0f, i};
            particles[i*num_particles_width + j].color = {0.2f, 0.2f, 0.6f};
            particles[i*num_particles_width + j].screen_pos = {0.0f, 0.0f, 0.0f};
            particles[i*num_particles_width + j].mass = 0.001;
        }
    }

    //create links for each spring

    //create vbo
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    unsigned size = get_num_particles() * sizeof(particle);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Cloth::render(float rotate_x, float rotate_y, float translate_z)
{
    render_particles(rotate_x, rotate_y, translate_z);
    render_springs(rotate_x, rotate_y, translate_z);
}

void Cloth::transform_particle_buffer()
{
    unsigned num_particles = get_num_particles();
    for(int i = 0; i < num_particles; i++)
    {
        particles[i].screen_pos.x = (float)particles[i].pos.x / num_particles_width;
        particles[i].screen_pos.y = 0;
        particles[i].screen_pos.z = (float)particles[i].pos.y / num_particles_height;

        particles[i].screen_pos.x = 2*particles[i].screen_pos.x - 1.0f;
        particles[i].screen_pos.z = 2*particles[i].screen_pos.z - 1.0f;

        particles[i].screen_pos.x = particles[i].pos.x;
        particles[i].screen_pos.z = particles[i].pos.z;
    }
}

void Cloth::render_particles(float rotate_x, float rotate_y, float translate_z)
{
    int num_particles = get_num_particles();
    transform_particle_buffer();

    for(int i = 0; i < num_particles; i++)
        printf("%d: %f %f %f \n", i, particles[i].screen_pos.x, particles[i].screen_pos.y, particles[i].screen_pos.z);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, num_particles * sizeof(particle), particles);
    glVertexPointer(3, GL_FLOAT, sizeof(particle), &particles[0]);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0f, 1.0f, 0.0f);
    glDrawArrays(GL_POINTS, 0, num_particles);
    glDisableClientState(GL_VERTEX_ARRAY);
}

inline unsigned Cloth::get_num_particles()
{
    return num_particles_width * num_particles_height;
}

inline unsigned Cloth::get_num_springs()
{
    return 0;
}

void Cloth::render_springs(float rotate_x, float rotate_y, float translate_z)
{
}

void Cloth::apply_forces()
{
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
    //TODO: FIXME
}

void Cloth::simulate_timestep()
{
    apply_forces();
    update_positions();
    satisfy_constraints();
    //TODO: FIXME
}
