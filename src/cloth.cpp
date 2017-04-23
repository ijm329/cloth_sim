#include "cloth.h"

Cloth::Cloth(int n)
{
    num_particles_width = n;
    num_particles_height = n;
    int num_particles = n * n;
    int num_shear = 2 * (num_particles_width - 1) * (num_particles_height - 1);
    int num_structural = num_particles_height * (num_particles_width - 1) + 
                         num_particles_width * (num_particles_height - 1);
    int num_flexion = num_particles_height * (num_particles_width - 2) + 
                       num_particles_width * (num_particles_height - 2);
    num_springs = num_shear + num_structural + num_flexion; 
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
    int num_shear = 2 * (num_particles_width - 1) * (num_particles_height - 1);
    int num_structural = num_particles_height * (num_particles_width - 1) + 
                         num_particles_width * (num_particles_height - 1);
    int num_flexion = num_particles_height * (num_particles_width - 2) + 
                       num_particles_width * (num_particles_height - 2);
    num_springs = num_shear + num_structural + num_flexion; 
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

// Dir: 0 ==> left 1 ==> right
void Cloth::make_diagonal_link(int i, int j, int &spring_cnt, int dir, float len)
{
    if(dir == LEFT)
    {
        springs[spring_cnt].left = ((i + 1) * num_particles_width) + j - 1;
        springs[spring_cnt].right = (i * num_particles_width) + j;
        ASSERT(springs[spring_cnt].left < get_num_particles());
    }
    else
    {
        springs[spring_cnt].left = (i * num_particles_width) + j;
        springs[spring_cnt].right = ((i + 1) * num_particles_width) + j + 1;
        ASSERT(springs[spring_cnt].right < get_num_particles());
    }
    springs[spring_cnt].rest_length = len;
    springs[spring_cnt].k = 1.0;
    springs[spring_cnt].damping = 1.0;
    springs[spring_cnt].spring_type = SHEAR;
    spring_cnt++;
    ASSERT(spring_cnt < num_springs);
}

void Cloth::make_structural_link(int i, int j, int target, int &spring_cnt, float len)
{
    springs[spring_cnt].left = (i * num_particles_width) + j;
    springs[spring_cnt].right = target;
    springs[spring_cnt].rest_length = len;
    springs[spring_cnt].k = 1.0;
    springs[spring_cnt].damping = 1.0;
    springs[spring_cnt].spring_type = STRUCTURAL;
    spring_cnt++;
    ASSERT(spring_cnt <= num_springs);
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
            particles[i*num_particles_width + j].color = {0.0f, 1.0f, 0.0f};
            particles[i*num_particles_width + j].mass = 0.001; //in kg
        }
    }
    float horizontal_length = 2.0f / ((float)num_particles_width);
    float vertical_length = 2.0f / ((float)num_particles_height);
    float diagonal_length = sqrtf(powf(horizontal_length, 2.0f) + 
                            powf(vertical_length, 2.0f));
    //create links for each spring
    int i, j;
    int spring_cnt = 0;
    //shear 
    for(i = 0; i < num_particles_height - 1; i++)
    {
        for(j = 0; j < num_particles_width; j++)
        {
            // If first then make diagonal towards the right 
            if(j == 0)
            {
                make_diagonal_link(i, j, spring_cnt, RIGHT, diagonal_length);
            }
            // If last then make diagonal towards the left
            else if(j == num_particles_width - 1)
            {
                make_diagonal_link(i, j, spring_cnt, LEFT, diagonal_length);
            }
            // In the middle, make both diagonals
            else
            {
                make_diagonal_link(i, j, spring_cnt, LEFT, diagonal_length);
                make_diagonal_link(i, j, spring_cnt, RIGHT, diagonal_length);
            }
        }
    }
    //structural 
    for(i = 0; i < num_particles_height; i++)
    {
        for(j = 0; j < num_particles_width; j++)
        {
            int right = (i * num_particles_width) + j + 1;
            int lower = ((i + 1) * num_particles_width) + j;
            if(j + 1 < num_particles_width)
            {
                make_structural_link(i, j, right, spring_cnt, horizontal_length);
            }
            if(i + 1 < num_particles_height)
            {
                make_structural_link(i, j, lower, spring_cnt, horizontal_length);
            }
        }
    }
    //flexion
    for(i = 0; i < num_particles_height; i++)
    {
        for(j = 0; j < num_particles_width; j++)
        {
            int right = (i * num_particles_width) + j + 2;
            int lower = ((i + 2) * num_particles_width) + j;
            //can use make_structural_link since it is essentially a structural 
            //link of length 2
            if(j + 2 < num_particles_width)
            {
                make_structural_link(i, j, right, spring_cnt, 2.0 * horizontal_length);
            }
            if(i + 2 < num_particles_height)
            {
                make_structural_link(i, j, lower, spring_cnt, 2.0 * vertical_length);
            }
        }
    }
    ASSERT(spring_cnt == num_springs);
    //create vbo
    //glGenBuffers(1, &vbo);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //unsigned size = get_num_particles() * sizeof(particle);
    //glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Cloth::render(float rotate_x, float rotate_y, float translate_z)
{
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


    //glBegin(GL_POINTS);
    //for(int i = 0; i < num_particles; i++)
    //{
    //    glColor3fv(&(particles[i].color.x));
    //    glVertex3fv(&(particles[i].screen_pos.x));
    //}
    //glEnd();

    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBufferSubData(GL_ARRAY_BUFFER, 0, num_particles * sizeof(particle), particles);
    //glVertexPointer(3, GL_FLOAT, sizeof(particle), &particles[0]);

    //glEnableClientState(GL_VERTEX_ARRAY);
    //glColor3f(0.0f, 1.0f, 0.0f);
    //glDrawArrays(GL_POINTS, 0, num_particles);
    //glDisableClientState(GL_VERTEX_ARRAY);
}

inline unsigned Cloth::get_num_particles()
{
    return num_particles_width * num_particles_height;
}

inline unsigned Cloth::get_num_springs()
{
    return num_springs;
}

void Cloth::render_springs(float rotate_x, float rotate_y, float translate_z)
{
    //rendering structural and shear springs only
    int num_flexion = num_particles_height * (num_particles_width - 2) + 
                       num_particles_width * (num_particles_height - 2);
    int num_shear_struct = num_springs - num_flexion;
    int i;
    vector3D start, end;
    glLineWidth(0.1);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    for(i = 0; i < num_shear_struct; i++)
    {
        start = particles[springs[i].left].pos;
        end = particles[springs[i].right].pos;
        glVertex3f(start.x, start.y, start.z);
        glVertex3f(end.x, end.y, end.z);
    }
    glEnd();
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
