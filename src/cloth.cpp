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

inline float get_stiffness(spring_type_t type)
{
    switch(type)
    {
        case STRUCTURAL:
            return K_STRUCT;
        case SHEAR:
            return K_SHEAR;
        case FLEXION:
            return K_FLEXION;
    }
}

inline float get_damping(spring_type_t type)
{
    switch(type)
    {
        case STRUCTURAL:
            return DAMPING_STRUCT;
        case SHEAR:
            return DAMPING_SHEAR;
        case FLEXION:
            return DAMPING_FLEXION;
    }
}

// Dir: 0 ==> left 1 ==> right
void Cloth::make_diagonal_link(int i, int j, int &spring_cnt, int dir, 
                               float len)
{
    if(dir == LEFT)
    {
        springs[spring_cnt].left = &(
                particles[((i + 1) * num_particles_width) + j - 1]);
        springs[spring_cnt].right = &(
                particles[(i * num_particles_width) + j]);
        //ASSERT(springs[spring_cnt].left < get_num_particles());
    }
    else
    {
        springs[spring_cnt].left = &(
                particles[(i * num_particles_width) + j]);
        springs[spring_cnt].right = &(
                particles[((i + 1) * num_particles_width) + j + 1]);
        //ASSERT(springs[spring_cnt].right < get_num_particles());
    }
    springs[spring_cnt].rest_length = len;
    springs[spring_cnt].k = get_stiffness(SHEAR);
    springs[spring_cnt].damping = get_damping(SHEAR);
    springs[spring_cnt].spring_type = SHEAR;
    spring_cnt++;
    ASSERT(spring_cnt < num_springs);
}

void Cloth::make_structural_link(int i, int j, int target, int &spring_cnt,
                                 float len, spring_type_t type)
{
    springs[spring_cnt].left = &(
            particles[(i * num_particles_width) + j]);
    springs[spring_cnt].right = &(particles[target]);
    springs[spring_cnt].rest_length = len;
    springs[spring_cnt].k = get_stiffness(type);
    springs[spring_cnt].damping = get_damping(type);
    springs[spring_cnt].spring_type = type;
    spring_cnt++;
    ASSERT(spring_cnt <= num_springs);
}

void Cloth::init()
{
    //cloth is a sheet in the xz plane

    //set positions of all particles
    for(int i = 0; i < num_particles_height; i++)
    {
        for(int j = 0; j < num_particles_width; j++)
        {
            float x,z;
            x = (float)j/num_particles_width;
            z = (float)i/num_particles_height;
            x = BOUND_LENGTH*x + MIN_BOUND;
            z = BOUND_LENGTH*z + MIN_BOUND;

            particles[i*num_particles_width + j].pos = {x, 1.0f, z};
            particles[i*num_particles_width + j].prev_pos = {x, 1.0f, z};

            //give the border particles a different color
            bool fixed;
            fixed = false;

            if(i == 0)
            {
                if((j == 0) || (j == num_particles_height - 1))
                    fixed = true;
            }

            particles[i*num_particles_width + j].fixed = fixed;
            particles[i*num_particles_width + j].force = {0.0f, 0.0f, 0.0f};
            particles[i*num_particles_width + j].normal = {0.0f, 0.0f, 0.0f};
        }
    }

    float horizontal_length = (BOUND_LENGTH) / ((float)num_particles_width);
    float vertical_length = (BOUND_LENGTH) / ((float)num_particles_height);
    float diagonal_length = sqrtf(POW_2(horizontal_length) + 
                            POW_2(vertical_length));

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
                make_diagonal_link(i, j, spring_cnt, RIGHT, diagonal_length);

            // If last then make diagonal towards the left
            else if(j == num_particles_width - 1)
                make_diagonal_link(i, j, spring_cnt, LEFT, diagonal_length);

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
                make_structural_link(i, j, right, spring_cnt, horizontal_length,
                                     STRUCTURAL);

            if(i + 1 < num_particles_height)
                make_structural_link(i, j, lower, spring_cnt,
                                     horizontal_length, STRUCTURAL);
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
                make_structural_link(i, j, right, spring_cnt, 2.0 * 
                                     horizontal_length, FLEXION);
            if(i + 2 < num_particles_height)
                make_structural_link(i, j, lower, spring_cnt, 
                                     2.0 * vertical_length, FLEXION);
        }
    }
    ASSERT(spring_cnt == num_springs);
}

void Cloth::reset_normals()
{
    for(int i = 0; i < get_num_particles(); i++)
        particles[i].normal = {0.0f, 0.0f, 0.0f};
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
    //render_springs(rotate_x, rotate_y, translate_z);
}

void Cloth::render_particles(float rotate_x, float rotate_y, float translate_z)
{
    int num_particles = get_num_particles();
    for(int i = 0; i < num_particles; i++)
        particles[i].normal.normalize();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(particle), &(particles[0].pos.x));
    glNormalPointer(GL_FLOAT, sizeof(particle), &(particles[0].normal.x));
    glColor3f(0.0f, 0.5f, 1.0f);
    glDrawArrays(GL_POINTS, 0, num_particles);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}

inline int Cloth::get_num_particles()
{
    return num_particles_width * num_particles_height;
}

inline int Cloth::get_num_springs()
{
    return num_springs;
}

//returns the vector normal to the triangle formed by these three points
vector3D Cloth::get_normal_vec(vector3D p1, vector3D p2, vector3D p3)
{
    vector3D line1 = p2 - p3;
    vector3D line2 = p3 - p1;
    vector3D cross_prod = line1.cross_product(line2);
    return cross_prod;
}

void Cloth::render_springs(float rotate_x, float rotate_y, float translate_z)
{
    //rendering structural and shear springs only
    int num_flexion = num_particles_height * (num_particles_width - 2) + 
                       num_particles_width * (num_particles_height - 2);
 
    int num_shear_struct = num_springs - num_flexion;

    //set to thinnest lines
    float line_width[2];
    glGetFloatv(GL_LINE_WIDTH_RANGE, line_width);
    glLineWidth(line_width[0]);

    /*glBegin(GL_LINES);
    for(int i = 0; i < num_shear_struct; i++)
    {
        glVertex3fv(&(springs[i].left->pos.x));
        glVertex3fv(&(springs[i].right->pos.x));
    }
    glEnd(); */
    //glColor3f(0.0, 0.90, 0.93);
}

void Cloth::apply_forces()
{
    //accumulate force of gravity on all particles
    vector3D gravity(0.0f, -9.81f, 0.0f);

    int num_particles = get_num_particles();
    for(int i = 0; i < num_particles; i++)
        particles[i].force = PARTICLE_MASS * gravity;

    apply_spring_forces();
    apply_wind_forces();
}

inline vector3D get_velocity(particle *p)
{
    return (p->pos - p->prev_pos) / TIME_STEP;
}

void Cloth::apply_spring_forces()
{
    int num_springs = get_num_springs();
    for(int i = 0; i < num_springs; i++)
    {
        particle *p1 = springs[i].left;
        particle *p2 = springs[i].right;

        //vector from left mass to right mass
        vector3D dir = (p2->pos) - (p1->pos);

        //rest vector is a the vector from left mass to right mass with
        //magnitude = length of spring
        vector3D rest = springs[i].rest_length * dir.unit();
        vector3D disp = dir - rest;
        vector3D vel = get_velocity(p2) - get_velocity(p1);

        vector3D spring_force = -springs[i].k*disp - springs[i].damping * vel;

        p1->force += -spring_force;
        p2->force += spring_force;
    }
}

void Cloth::apply_wind_forces()
{
    //calculate vector normal to the points on the mesh to account for 
    //wind force. Make two triangles that form a square that travels around the 
    //mesh computing the wind forces
    for(int j = 0; j < num_particles_height - 1; j++)
    {
        for(int i = 0; i < num_particles_width - 1; i++)
        {
            int curr_idx = j * num_particles_width + i;
            int right_idx = curr_idx + 1;
            int lower_idx = (j + 1) * num_particles_width + i;
            int diag_idx = lower_idx + 1;

            vector3D curr = particles[curr_idx].pos;
            vector3D right = particles[right_idx].pos;
            vector3D lower = particles[lower_idx].pos;
            vector3D diagonal = particles[diag_idx].pos;

            // make two triangles to complete a square 
            vector3D wind(WIND_X, WIND_Y, WIND_Z);
            vector3D norm1 = get_normal_vec(right, curr, lower);
            vector3D norm2 = get_normal_vec(diagonal, right, lower);

            //add norm1 to particles
            particles[right_idx].normal += norm1;
            particles[curr_idx].normal += norm1;
            particles[lower_idx].normal += norm1;

            //add norm2 to particles
            particles[diag_idx].normal += norm2;
            particles[right_idx].normal += norm2;
            particles[lower_idx].normal += norm2;

            vector3D wind_force1 = norm1 * (norm1.unit().dot_product(wind));
            vector3D wind_force2 = norm2 * (norm2.unit().dot_product(wind));
            
            //force 1
            particles[right_idx].force += wind_force1;
            particles[curr_idx].force += wind_force1;
            particles[lower_idx].force += wind_force1;

            //force2
            particles[right_idx].force += wind_force2;
            particles[diag_idx].force += wind_force2;
            particles[lower_idx].force += wind_force2;
        }
    }
}

void Cloth::reset_fixed_particles()
{
    for(int i = 0; i < num_particles_height; i++)
    {
        for(int j = 0; j < num_particles_width; j++)
        {
            int index = i*num_particles_width + j;
            if(particles[index].fixed)
            {
                float x = (float)j/num_particles_width;
                float z = (float)i/num_particles_height;
                x = BOUND_LENGTH*x + MIN_BOUND;
                z = BOUND_LENGTH*z + MIN_BOUND;

                particles[index].pos = {x, 1.0f, z};
            }
        }
    }
}

void Cloth::satisfy_constraints()
{
    int num_shear = 2 * (num_particles_width - 1) * (num_particles_height - 1);
    int num_structural = num_particles_height * (num_particles_width - 1) + 
                         num_particles_width * (num_particles_height - 1);
    for(int k = 0; k < NUM_CONSTRAINT_ITERS; k++)
    {
        for(int i = 0; i < num_shear + num_structural; i++)
        {
            particle *p1 = springs[i].left;
            particle *p2 = springs[i].right;
            vector3D diff = (p2->pos) - (p1->pos);

            float new_length = diff.norm();
            if(new_length > (STRETCH_CRITICAL*springs[i].rest_length))
            {
                float move_dist = (new_length - 
                        (STRETCH_CRITICAL*springs[i].rest_length))/2.0;
                if(!p1->fixed && !p2->fixed)
                {
                    p1->pos += move_dist * diff.unit();
                    p2->pos -= move_dist * diff.unit();
                }
                else if(!p1->fixed)
                    p1->pos += 2*move_dist * diff.unit();
                else if(!p2->fixed)
                    p2->pos -= 2*move_dist * diff.unit();
            }
        }
        reset_fixed_particles();
    }
}

void Cloth::update_positions()
{
    //perform verlet integration
    int num_particles = get_num_particles();
    //std::cout<<"Forces " << std::endl;
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
    double reset_start = CycleTimer::currentSeconds();
    reset_normals();
    double reset_end = CycleTimer::currentSeconds();

    double forces_start = CycleTimer::currentSeconds();
    apply_forces();
    double forces_end = CycleTimer::currentSeconds();

    double update_start = CycleTimer::currentSeconds();
    update_positions();
    double update_end = CycleTimer::currentSeconds();

    double constraint_start = CycleTimer::currentSeconds();
    satisfy_constraints();
    double constraint_end = CycleTimer::currentSeconds();

    printf("----------------------------------------\n");
    printf("Reset Normals : %.3f s \n", reset_end - reset_start);
    printf("Apply Forces : %.3f s \n", forces_end-forces_start);
    printf("Update Positions : %.3f s \n", update_end-update_start);
    printf("Satisfy Constraints : %.3f s \n", constraint_end-constraint_start);
}
