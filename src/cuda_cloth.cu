#include "cuda_cloth.h"

#define POS_INDEX 0
#define PREV_POS_INDEX 1

extern GLuint pos_vbo, normal_vbo;
extern struct cudaGraphicsResource *cuda_pos_vbo_resource;
extern struct cudaGraphicsResource *cuda_normal_vbo_resource;

struct cloth_constants
{
    int num_particles_width;
    int num_particles_height;
    float struct_spring_len;
    float shear_spring_len;
    float3 *prev_pos_array;
    float3 *force_array;
};

typedef struct
{
    float3 pos;
    float3 prev_pos;
} particle_pos_data_t;

__constant__ cloth_constants cuda_cloth_params;

cuda_cloth::cuda_cloth(int n)
{
    num_particles_width = n;
    num_particles_height = n;
    num_particles = num_particles_width * num_particles_height;
    host_pos_array = (float3 *)malloc(sizeof(float3) * num_particles);
    if((host_pos_array == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    GPU_ERR_CHK(cudaMalloc(&dev_prev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_force_array, sizeof(float3) * num_particles));
}

cuda_cloth::cuda_cloth(int w, int h)
{
    num_particles_width = w;
    num_particles_height = h;
    host_pos_array = (float3 *)malloc(sizeof(float3) * num_particles);
    if((host_pos_array == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    GPU_ERR_CHK(cudaMalloc(&dev_prev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_force_array, sizeof(float3) * num_particles));
}

cuda_cloth::~cuda_cloth()
{
    cudaFree(dev_prev_pos_array);
    cudaFree(dev_force_array);
}

__inline__ float get_spring_len(int width, int height, spring_type_t type)
{
     if(type == STRUCTURAL)
        return (BOUND_LENGTH) / ((float)width);
    //Shear
    else
    { 
        float h_len = (BOUND_LENGTH) / ((float)width);
        float v_len = (BOUND_LENGTH) / ((float)height);
        return sqrtf(POW_2(h_len) + POW_2(v_len));
    }
}

void create_vbo(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
                unsigned int vbo_res_flags, unsigned num_particles)
{
    assert(vbo);

    //create the buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    //initialize buffer object
    unsigned int size = num_particles * sizeof(float3);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

size_t get_cuda_device_ptr(struct cudaGraphicsResource **vbo_resource,
                             float3 **dptr)
{
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)dptr,
                                    &num_bytes, *vbo_resource));
    return num_bytes;
}

// initializes cloth positions and allocates memory region in the device and 
// copies the particle buffer into that region
void cuda_cloth::init()
{
    for(int i = 0; i < num_particles_height; i++)
    {
        for(int j = 0; j < num_particles_width; j++)
        {
            float x,z;
            x = (float)j/num_particles_width;
            z = (float)i/num_particles_height;
            x = BOUND_LENGTH*x + MIN_BOUND;
            z = BOUND_LENGTH*z + MIN_BOUND;

            host_pos_array[i*num_particles_width + j] = {x, 1.0f, z};
        }
    }
    //clear forces
    GPU_ERR_CHK(cudaMemset(dev_force_array, 0, sizeof(float3) * num_particles));

    //set up cloth simulation parameters in read only memory
    cloth_constants params;
    params.num_particles_width = num_particles_width;
    params.num_particles_height = num_particles_height;
    params.prev_pos_array = dev_prev_pos_array;
    //params.normal_array = dev_normal_array;
    params.force_array = dev_force_array;
    params.struct_spring_len = get_spring_len(num_particles_width,
                               num_particles_height, STRUCTURAL);
    params.shear_spring_len = get_spring_len(num_particles_width,
                              num_particles_height, SHEAR);
    GPU_ERR_CHK(cudaMemcpyToSymbol(cuda_cloth_params, &params,
                                  sizeof(cloth_constants)));
    create_vbo(&pos_vbo, &cuda_pos_vbo_resource, cudaGraphicsMapFlagsNone,
               num_particles_width * num_particles_height);
    create_vbo(&normal_vbo, &cuda_normal_vbo_resource, 
               cudaGraphicsMapFlagsWriteDiscard,
               num_particles_width * num_particles_height);
    float3 *dptr, *nptr;
    //copy over data into buffers and clear as necessary
    size_t num_bytes = get_cuda_device_ptr(&cuda_pos_vbo_resource, &dptr);
    size_t norm_bytes = get_cuda_device_ptr(&cuda_normal_vbo_resource, &nptr);
    GPU_ERR_CHK(cudaMemcpy(dptr, host_pos_array, num_bytes, cudaMemcpyHostToDevice));
    GPU_ERR_CHK(cudaMemcpy(dev_prev_pos_array, dptr, num_bytes, cudaMemcpyDeviceToDevice));
    GPU_ERR_CHK(cudaMemset(nptr, 0, sizeof(float3) * num_particles));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_vbo_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_normal_vbo_resource, 0));
    free(host_pos_array);
    host_pos_array = NULL;
}


void cuda_cloth::render(float rotate_x, float rotate_y, float translate_z)
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_vbo_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_normal_vbo_resource,
                                               0));
    unsigned num_elements = num_particles; 

    //transform the cloth's position in the world space based on 
    //camera parameters
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo);
    glNormalPointer(GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glColor3f(0.0, 1.0, 0.5);
    glDrawArrays(GL_POINTS, 0, num_elements);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}

__device__ __inline__ float3 get_velocity(float3 p_pos, float3 p_prev_pos)
{
    return (p_pos - p_prev_pos) * (1.0f / TIME_STEP);
}

__device__ __inline__ float get_spring_const(spring_type_t type)
{
    if(type == STRUCTURAL)
        return K_STRUCT;
    else if(type == SHEAR)
        return K_SHEAR;
    else 
        return K_FLEXION;
}

__device__ __inline__ float get_spring_damp(spring_type_t type)
{
    if(type == STRUCTURAL)
        return DAMPING_STRUCT;
    else if(type == SHEAR)
        return DAMPING_SHEAR;
    else 
        return DAMPING_FLEXION;
}

__device__ float3 compute_wind_force(float3 p1, float3 p2, float3 p3, float3 *p_norm)
{
    float3 line1 = p2 - p3;
    float3 line2 = p3 - p1;

    //normal to the triangle
    float3 norm = cross(line1, line2);
    *(p_norm) = *(p_norm) + norm;
    float3 wind = make_float3(WIND_X, WIND_Y, WIND_Z);
    float3 wind_force = norm * (dot(normalize(norm), wind));
    return wind_force; 
}

__device__ float3 compute_spring_force(float3 p1_pos, float3 p2_pos,
                                       float3 p1_prev_pos, 
                                       float3 p2_prev_pos, 
                                       float len, spring_type_t type)
{
    float3 dir = p2_pos - p1_pos;
    float3 rest = len * (normalize(dir));
    float spr_const = get_spring_const(type);
    float damp_const = get_spring_damp(type);
    float3 disp = (spr_const * dir) - (spr_const * rest);
    float3 vel = get_velocity(p2_pos, p2_prev_pos) - get_velocity(p1_pos, 
                                                               p1_prev_pos);

    float3 force = -disp - (damp_const * vel);
    return -force;
}

__device__ __inline__ void load_particle_pos_data(
        particle_pos_data_t blk_particles[][TPB_X+2],
        float3 *f_top[2],
        float3 *f_btm[2],
        float3 *f_left[2],
        float3 *f_right[2],
        float3 *dptr)
{
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    float3 *dev_pos_array = dptr;
    float3 *dev_prev_pos_array = cuda_cloth_params.prev_pos_array;

    int shared_mem_x =  blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;

    //supeblock(TPB_y+2, TPB_x+2)
    //sblk_row and sblk_col are indices in the super block
    int sblk_row = threadIdx.y + 1;
    int sblk_col = threadIdx.x + 1;
    int idx = (row * width) + col;

    if(row < height && col < width)
    {
        //cooperatively load into the array the necessary particles starting 
        //with those contained in the thread block.
        blk_particles[sblk_row][sblk_col].pos = dev_pos_array[idx];
        blk_particles[sblk_row][sblk_col].prev_pos = dev_prev_pos_array[idx];
        
        
        //for border particles, load the immediate top, left, bottom, and right
        //load tops so only if you're in the top row of your block

        //top particles
        if(threadIdx.y == 0)
        {
            //ensure you're not the top row in the system
            if(row != 0)
            {
                int index = (row - 1) * width + col;
                blk_particles[0][sblk_col].pos = dev_pos_array[index];
                blk_particles[0][sblk_col].prev_pos = 
                                        dev_prev_pos_array[index];
            }
            else
                blk_particles[0][sblk_col].pos = make_float3(HUGE_VALF,
                                                HUGE_VALF, HUGE_VALF);
        }

        //bottom particles
        if(threadIdx.y == blockDim.y - 1)
        {
            //not the bottom row in the system
            if(row != height - 1)
            {
                int index = (row + 1) * width + col;
                blk_particles[sblk_row + 1][sblk_col].pos = 
                                                    dev_pos_array[index];
                blk_particles[sblk_row + 1][sblk_col].prev_pos =
                                                    dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row + 1][sblk_col].pos = make_float3(
                                    HUGE_VALF, HUGE_VALF, HUGE_VALF);
        }

        //left particles
        if(threadIdx.x == 0)
        {
            //if you're not the left edge in the system 
            if(col != 0)
            {
                int index = idx-1;
                blk_particles[sblk_row][0].pos = dev_pos_array[index];
                blk_particles[sblk_row][0].prev_pos = 
                                            dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row][0].pos = make_float3(HUGE_VALF,
                                                HUGE_VALF, HUGE_VALF);
        }

        //right particles
        if(threadIdx.x == blockDim.x - 1)
        {
            //if you're not the right edge in the system
            if(col != width - 1)
            {
                int index = idx+1;
                blk_particles[sblk_row][sblk_col + 1].pos = 
                                                    dev_pos_array[index];
                blk_particles[sblk_row][sblk_col + 1].prev_pos =
                                                    dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row][sblk_col + 1].pos = make_float3(
                                    HUGE_VALF, HUGE_VALF, HUGE_VALF);
        }

        //corners 
        //top left
        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            //if not at the top of the system and not at the top left edge
            if(row != 0 && col != 0)
            {
                int index = (row - 1) * width + (col - 1);
                blk_particles[0][0].pos = dev_pos_array[index];
                blk_particles[0][0].prev_pos = dev_prev_pos_array[index];
            }
            else
                blk_particles[0][0].pos = make_float3(HUGE_VALF, 
                        HUGE_VALF, HUGE_VALF);
        }

        //top right
        if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
        {
            if(row != 0 && col != width - 1)
            {
                int index = (row - 1) * width + (col + 1);
                blk_particles[0][shared_mem_x - 1].pos = dev_pos_array[index];
                blk_particles[0][shared_mem_x - 1].prev_pos = 
                                                    dev_prev_pos_array[index];
            }
            else
                blk_particles[0][shared_mem_x - 1].pos = make_float3(HUGE_VALF,
                                                        HUGE_VALF, HUGE_VALF);
        }

        //bottom left
        if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != 0)
            {
                int index = (row + 1) * width + (col - 1);
                blk_particles[shared_mem_y - 1][0].pos = dev_pos_array[index];
                blk_particles[shared_mem_y - 1][0].prev_pos = 
                                                    dev_prev_pos_array[index];
            }
            else
                blk_particles[shared_mem_y - 1][0].pos = make_float3(HUGE_VALF,
                                                        HUGE_VALF, HUGE_VALF);
        }

        //bottom right 
        if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != width - 1)
            {
                int index = (row + 1) * width + (col + 1);
                blk_particles[shared_mem_y - 1][shared_mem_x - 1].pos =
                                                        dev_pos_array[index];
                blk_particles[shared_mem_y - 1][shared_mem_x - 1].prev_pos =
                                                    dev_prev_pos_array[index];
            }
            else 
                blk_particles[shared_mem_y - 1][shared_mem_x - 1].pos =
                            make_float3(HUGE_VALF, HUGE_VALF, HUGE_VALF);
        }

        //set the flexion springs

        //ensure that you have an upper flexion spring 
        if((row - 2) >= 0)
        {
            int f_top_id = (row - 2) * width + col;
            if((sblk_row - 2) >= 0)
            {
                f_top[POS_INDEX] = &blk_particles[sblk_row-2][sblk_col].pos;
                f_top[PREV_POS_INDEX] = &(
                        blk_particles[sblk_row-2][sblk_col].prev_pos);
            }
            else
            {
                f_top[POS_INDEX] = &dev_pos_array[f_top_id];
                f_top[PREV_POS_INDEX] = &dev_prev_pos_array[f_top_id];
            }
        }

        if((row + 2) < height)
        {
            int f_btm_id = (row + 2) * width + col;
            if((sblk_row + 2) < shared_mem_y)
            {
                f_btm[POS_INDEX] = &blk_particles[sblk_row + 2][sblk_col].pos;
                f_btm[PREV_POS_INDEX] = &(
                        blk_particles[sblk_row + 2][sblk_col].prev_pos);
            }
            else
            {
                f_btm[POS_INDEX] = &dev_pos_array[f_btm_id];
                f_btm[PREV_POS_INDEX] = &dev_prev_pos_array[f_btm_id];
            }
        }

        if((col - 2) >= 0)
        {
            int f_left_id = idx - 2; 
            if((sblk_col - 2) >= 0)
            {
                f_left[POS_INDEX] = &blk_particles[sblk_row][sblk_col - 2].pos;
                f_left[PREV_POS_INDEX] = &(
                        blk_particles[sblk_row][sblk_col - 2].prev_pos);
            }
            else
            {
                f_left[POS_INDEX] = &dev_pos_array[f_left_id];
                f_left[PREV_POS_INDEX] = &dev_prev_pos_array[f_left_id];
            }
        }
        
        if((col + 2) < width)
        {
            int f_right_id = idx + 2;
            if((sblk_col + 2) < shared_mem_x)
            {
                f_right[POS_INDEX] = &blk_particles[sblk_row][sblk_col + 2].pos;
                f_right[PREV_POS_INDEX] = &(
                        blk_particles[sblk_row][sblk_col + 2].prev_pos);
            }
            else
            {
                f_right[POS_INDEX] = &dev_pos_array[f_right_id];
                f_right[PREV_POS_INDEX] = &dev_prev_pos_array[f_right_id];
            }
        }
    }
}

__device__ __inline__ float3 compute_particle_forces(
        particle_pos_data_t blk_particles[][TPB_X+2],
        float3 *f_top[2],
        float3 *f_btm[2],
        float3 *f_left[2],
        float3 *f_right[2], 
        float3 *norm_arr)
{
    float3 tot_force = make_float3(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    float struct_len = cuda_cloth_params.struct_spring_len;
    float shear_len = cuda_cloth_params.shear_spring_len;
    float flex_len = 2.0f * struct_len;
    
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;

    int sblk_row = threadIdx.y + 1;
    int sblk_col = threadIdx.x + 1;

    float3 particle_normal = make_float3(0.0f, 0.0f, 0.0f);

    //Now that all necessary particles have been loaded, compute wind forces
    //computing wind forces and spring forces
    if(row < height && col < width)
    {
        int idx = row * width + col;
        //your row and col within the shared mem matrix
        particle_pos_data_t curr = blk_particles[sblk_row][sblk_col];
        if(row != 0 && col != width - 1)
        {
            particle_pos_data_t top = blk_particles[sblk_row - 1][sblk_col];
            particle_pos_data_t top_right = 
                blk_particles[sblk_row-1][sblk_col+1];
            tot_force += compute_wind_force(top_right.pos, top.pos, curr.pos,
                                            &particle_normal);
            particle_pos_data_t right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, top_right.pos, curr.pos,
                                            &particle_normal);
            tot_force += compute_spring_force(curr.pos, top_right.pos, 
                         curr.prev_pos, top_right.prev_pos, shear_len, SHEAR);
        }
        if(row != height - 1 && col != width - 1)
        {
            particle_pos_data_t bottom = blk_particles[sblk_row + 1][sblk_col];
            particle_pos_data_t right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, curr.pos, bottom.pos,
                                            &particle_normal);
            particle_pos_data_t bottom_right =
                blk_particles[sblk_row + 1][sblk_col + 1];
            tot_force += compute_spring_force(curr.pos, bottom_right.pos, 
                         curr.prev_pos, bottom_right.prev_pos, shear_len, 
                         SHEAR);
        }
        if(row != height - 1 && col != 0)
        {
            particle_pos_data_t bottom = blk_particles[sblk_row + 1][sblk_col];
            particle_pos_data_t bottom_left =
                        blk_particles[sblk_row + 1][sblk_col - 1];
            tot_force += compute_wind_force(bottom.pos, curr.pos, 
                                            bottom_left.pos, &particle_normal);
            particle_pos_data_t left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, left.pos, 
                                            bottom_left.pos, &particle_normal);
            tot_force += compute_spring_force(curr.pos, bottom_left.pos, 
                        curr.prev_pos, bottom_left.prev_pos, shear_len, SHEAR);
        }
        if(row != 0 && col != 0)
        {
            particle_pos_data_t top = blk_particles[sblk_row - 1][sblk_col];
            particle_pos_data_t left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, top.pos, left.pos,
                                            &particle_normal);
            particle_pos_data_t top_left =
                        blk_particles[sblk_row - 1][sblk_col - 1];
            tot_force += compute_spring_force(curr.pos, top_left.pos, 
                        curr.prev_pos, top_left.prev_pos, shear_len, SHEAR);
        }

        particle_normal = normalize(particle_normal);
        norm_arr[idx] = particle_normal;

        //structural forces
        if(col != 0)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row][sblk_col - 1].pos, curr.prev_pos,
              blk_particles[sblk_row][sblk_col - 1].prev_pos, struct_len, 
              STRUCTURAL);
        if(col != width - 1)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row][sblk_col + 1].pos, curr.prev_pos,
              blk_particles[sblk_row][sblk_col + 1].prev_pos, struct_len, 
              STRUCTURAL);
        if(row != 0)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row - 1][sblk_col].pos, curr.prev_pos,
              blk_particles[sblk_row - 1][sblk_col].prev_pos, struct_len, 
              STRUCTURAL);
        if(row != height - 1)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row + 1][sblk_col].pos, curr.prev_pos,
              blk_particles[sblk_row + 1][sblk_col].prev_pos, struct_len, 
              STRUCTURAL);

        //flexion forces 
        if(f_top[POS_INDEX]) 
            tot_force += compute_spring_force(curr.pos, *(f_top[POS_INDEX]), 
                    curr.prev_pos, *(f_top[PREV_POS_INDEX]), flex_len, FLEXION);
        if(f_btm[POS_INDEX]) 
            tot_force += compute_spring_force(curr.pos, *(f_btm[POS_INDEX]),
                    curr.prev_pos, *(f_btm[PREV_POS_INDEX]), flex_len, FLEXION);
        if(f_left[POS_INDEX]) 
            tot_force += compute_spring_force(curr.pos, *(f_left[POS_INDEX]),
                curr.prev_pos, *(f_left[PREV_POS_INDEX]), flex_len, FLEXION);
        if(f_right[POS_INDEX]) 
            tot_force += compute_spring_force(curr.pos, *(f_right[POS_INDEX]),
                curr.prev_pos, *(f_right[PREV_POS_INDEX]), flex_len, FLEXION);
    }
    return tot_force;
}

__global__ void apply_forces_kernel(float3 *dptr, float3 *nptr)
{
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;

    int idx = (row * width) + col;
    float3 *dev_force_array = cuda_cloth_params.force_array;

    //create a 2D array of shared memory for particles that will be used by 
    //block
    __shared__ particle_pos_data_t blk_particles[TPB_Y+2][TPB_X+2];

    float3 *f_top[2] = {NULL, NULL};
    float3 *f_btm[2] = {NULL, NULL};
    float3 *f_left[2] = {NULL, NULL};
    float3 *f_right[2] = {NULL, NULL};

    load_particle_pos_data(blk_particles, f_top, f_btm, f_left, f_right, dptr);
    __syncthreads();
    float3 tot_force = compute_particle_forces(blk_particles, f_top, f_btm,
                                               f_left, f_right, nptr);
    if((row < height) && (col < width))
      dev_force_array[idx] = tot_force;
}

void cuda_cloth::apply_forces(float3 *dptr, float3 *nptr)
{
  //setup invocaiton for application of all forces
  dim3 threads_per_block(TPB_X, TPB_Y, 1);
  dim3 num_blocks(UP_DIV(num_particles_width, TPB_X), 
                 UP_DIV(num_particles_height, TPB_Y), 1);
  apply_forces_kernel<<<num_blocks, threads_per_block>>>(dptr, nptr);
  cudaDeviceSynchronize();
}

__global__ void update_positions_kernel(float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //if valid particle index
    if(row < height && col < width)
    {
        int i = row * width + col;
     
        float3 curr_pos = dptr[i];
        float3 curr_prev_pos = cuda_cloth_params.prev_pos_array[i];
        float3 curr_force = cuda_cloth_params.force_array[i];
        float3 temp = make_float3(curr_pos.x, curr_pos.y, curr_pos.z);
        float3 acc = curr_force/PARTICLE_MASS;
        curr_pos += (curr_pos - curr_prev_pos + acc * TIME_STEP * TIME_STEP);
        curr_prev_pos = temp;
        dptr[i] = curr_pos;
        cuda_cloth_params.prev_pos_array[i] = curr_prev_pos;
        
        //fixed particles
        if((row == 0 && col == 0) || (row == 0 && col == width - 1))
        {
          float x = (float)col/width;
          float  z = (float)row/height;
          x = BOUND_LENGTH*x + MIN_BOUND;
          z = BOUND_LENGTH*z + MIN_BOUND;
          dptr[i] = make_float3(x,1.0f,z);
        }
    }
}

void cuda_cloth::update_positions(float3 *dptr)
{
    dim3 threads_per_block(TPB_X, TPB_Y, 1);
    dim3 num_blocks(UP_DIV(num_particles_width, TPB_X), 
                   UP_DIV(num_particles_height, TPB_Y), 1);
    update_positions_kernel<<<num_blocks, threads_per_block>>>(dptr);
    cudaDeviceSynchronize();
}

__device__ __inline__ bool is_fixed(int row, int col)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    assert((row>=0) && (row<height) && (col>=0) && (col<width));
    return (((row == 0) && (col == 0)) || (row == 0) && (col == width-1));
}

__device__ __inline__ void satisfy_constraint(float3 *pos1, float3 *pos2, 
                            float rest_len, bool p1_fixed, bool p2_fixed)
{
    float3 diff = *pos2 - *pos1;
    float new_length = length(diff);
    if(new_length > (STRETCH_CRITICAL*rest_len))
    {
        float move_dist = (new_length - (STRETCH_CRITICAL*rest_len))/2.0;
        if(!p1_fixed && !p2_fixed)
        {
            *(pos1) = *(pos1) + move_dist * normalize(diff);
            *(pos2) = *(pos2) - move_dist * normalize(diff);
        }
        else if(!p1_fixed)
            *(pos1) = *(pos1) + 2*move_dist * normalize(diff);
        else if(!p2_fixed)
            *(pos2) = *(pos2) - 2*move_dist * normalize(diff);
    }
}
__device__ __inline__ void satisfy_six_constraints(int row, int col, float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    float struct_len = cuda_cloth_params.struct_spring_len;
    float shear_len = cuda_cloth_params.shear_spring_len;
    float3 *pos_array = dptr;

    assert((row <= (height-2)) && (col <= (width-2)));
    float3 *curr = &pos_array[row*width+col];
    float3 *right = &pos_array[row*width+col+1];
    float3 *bottom = &pos_array[(row+1)*width+col];
    float3 *bottom_right = &pos_array[(row+1)*width+col+1];

    bool curr_fixed = is_fixed(row, col);
    bool right_fixed = is_fixed(row, col+1);
    bool bot_fixed = is_fixed(row+1, col);
    bool bot_right_fixed = is_fixed(row+1, col+1);

    satisfy_constraint(curr, right, struct_len, curr_fixed, right_fixed);
    satisfy_constraint(right, bottom_right, struct_len, right_fixed, bot_fixed);
    satisfy_constraint(bottom_right, bottom, struct_len, bot_right_fixed, 
                       bot_fixed);
    satisfy_constraint(bottom, curr, struct_len, bot_fixed, curr_fixed);
    satisfy_constraint(curr, bottom_right, shear_len, curr_fixed, 
                       bot_right_fixed);
    satisfy_constraint(bottom, right, shear_len, bot_fixed, right_fixed);
}

__global__ void satisfy_case1_kernel(float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int num_constraint_groups = (width/2) * (height/2);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_constraint_groups)
    {
        int cg_row = idx / (width/2);
        int cg_col = idx % (width/2);
        int top_left_row = cg_row*2;
        int top_left_col = cg_col*2;
        satisfy_six_constraints(top_left_row, top_left_col, dptr);
    }
}

__global__ void satisfy_case2_kernel(float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int num_constraint_groups = (height/2) * ((width-1)/2);
    int len = (width-1)/2;
    float3 *pos_array = dptr;
    float shear_len = cuda_cloth_params.shear_spring_len;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_constraint_groups)
    {
        int cg_row = idx / len;
        int cg_col = idx % len;
        int row = cg_row*2;
        int col = cg_col*2+1;

        float3 *curr = &pos_array[row*width+col];
        float3 *right = &pos_array[row*width+col+1];
        float3 *bottom = &pos_array[(row+1)*width+col];
        float3 *bottom_right = &pos_array[(row+1)*width+col+1];

        bool curr_fixed = is_fixed(row, col);
        bool right_fixed = is_fixed(row, col+1);
        bool bot_fixed = is_fixed(row+1, col);
        bool bot_right_fixed = is_fixed(row+1, col+1);

        satisfy_constraint(curr, bottom_right, shear_len, curr_fixed, 
                           bot_right_fixed);
        satisfy_constraint(bottom, right, shear_len, bot_fixed, right_fixed);
    }
}

__global__ void satisfy_case3_kernel(float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int num_constraint_groups = (width/2) * ((height-1)/2);
    int len = (width/2);
    float3 *pos_array = dptr;
    float shear_len = cuda_cloth_params.shear_spring_len;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_constraint_groups)
    {
        int cg_row = idx / len;
        int cg_col = idx % len;
        int row = cg_row*2+1;
        int col = cg_col*2;

        float3 *curr = &pos_array[row*width+col];
        float3 *right = &pos_array[row*width+col+1];
        float3 *bottom = &pos_array[(row+1)*width+col];
        float3 *bottom_right = &pos_array[(row+1)*width+col+1];

        bool curr_fixed = is_fixed(row, col);
        bool right_fixed = is_fixed(row, col+1);
        bool bot_fixed = is_fixed(row+1, col);
        bool bot_right_fixed = is_fixed(row+1, col+1);

        satisfy_constraint(curr, bottom_right, shear_len, curr_fixed, 
                           bot_right_fixed);
        satisfy_constraint(bottom, right, shear_len, bot_fixed, right_fixed);
    }
}

__global__ void satisfy_case4_kernel(float3 *dptr)
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int num_constraint_groups = ((width-2)/2) * ((height-2)/2) + (width-2) + 
                                (height-2);
    float3 *pos_array = dptr;
    float struct_len = cuda_cloth_params.struct_spring_len;

    int num_inner_square_constraints = ((width-2)/2) * ((height-2)/2);
    int num_row_border_constraints = (width-2);
    int num_col_border_constraints = (width-2);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_constraint_groups)
    {
        //subcase 1 inner square
        if(idx < (num_inner_square_constraints))
        {
            int len = (width-2)/2;
            int cg_row = idx / len;
            int cg_col = idx % len;
            int row = cg_row*2+1;
            int col = cg_col*2+1;
            satisfy_six_constraints(row, col, dptr);
        }
        //subcase 2 border rows
        else if(idx < (num_inner_square_constraints + 
                       num_row_border_constraints))
        {
            int new_idx = idx - num_inner_square_constraints;
            int len = num_row_border_constraints/2;
            int row = 0;
            if(new_idx > len)
                row = height-1;
            else
                row = 0;
            int col = new_idx % len;

            float3 *curr = &pos_array[row*width+col];
            float3 *right = &pos_array[row*width+col+1];

            bool curr_fixed = is_fixed(row, col);
            bool right_fixed = is_fixed(row, col+1);
            satisfy_constraint(curr, right, struct_len, curr_fixed, 
                               right_fixed);
        }
        //subcase 3 border columns
        else
        {
            int new_idx = (idx - num_inner_square_constraints -
                                 num_row_border_constraints);
            int len = num_col_border_constraints/2;
            int col = 0;
            if(new_idx > len)
                col = width-1;
            else
                col = 0;
            int row = new_idx % len;

            float3 *curr = &pos_array[row*width+col];
            float3 *bot = &pos_array[(row+1)*width+col];

            bool curr_fixed = is_fixed(row, col);
            bool bot_fixed = is_fixed(row+1, col);
            satisfy_constraint(curr, bot, struct_len, curr_fixed, bot_fixed);
        }
    }
}

void cuda_cloth::satisfy_constraints(float3 *dptr)
{
    int threads_per_block = 1024;

    for(int i = 0; i < NUM_CONSTRAINT_ITERS; i++)
    {
        //case 1
        // o----o   o----o
        // |\  /|   |\  /|
        // | \/ |   | \/ |
        // | /\ |   | /\ |
        // |/  \|   |/  \|
        // o----o   o----o
        // o----o   o----o
        // |\  /|   |\  /|
        // | \/ |   | \/ |
        // | /\ |   | /\ |
        // |/  \|   |/  \|
        // o----o   o----o
        int width = num_particles_width;
        int height = num_particles_height;
        int num_constraint_groups = (width/2) * (height/2);
        int num_blocks = UP_DIV(num_constraint_groups, threads_per_block);
        satisfy_case1_kernel<<<num_blocks, threads_per_block>>>(dptr);
        cudaDeviceSynchronize();

        //case 2
        // o   o   o   o   o
        //      \ /     \ / 
        //       \       \
        //      / \     / \
        // o   o   o   o   o
        // o   o   o   o   o
        //      \ /     \ / 
        //       \       \
        //      / \     / \
        // o   o   o   o   o
        num_constraint_groups = (height/2) * ((width-1)/2);
        num_blocks = UP_DIV(num_constraint_groups, threads_per_block);
        satisfy_case2_kernel<<<num_blocks, threads_per_block>>>(dptr);
        cudaDeviceSynchronize();

        //case 3
        // o  o  o  o  o  o
        //
        // o  o  o  o  o  o
        //  \/    \/    \/
        //  /\    /\    /\
        // o  o  o  o  o  o
        // o  o  o  o  o  o
        //  \/    \/    \/
        //  /\    /\    /\
        // o  o  o  o  o  o
        num_constraint_groups = (width/2) * ((height-1)/2);
        num_blocks = UP_DIV(num_constraint_groups, threads_per_block);
        satisfy_case3_kernel<<<num_blocks, threads_per_block>>>(dptr);
        cudaDeviceSynchronize();

        //case4
        // o  o--o  o--o  o
        //
        // o  o--o  o--o  o
        // |  |\/|  |\/|  |
        // |  |/\|  |/\|  |
        // o  o--o  o--o  o
        //
        // o  o--o  o--o  o
        // |  |\/|  |\/|  |
        // |  |/\|  |/\|  |
        // o  o--o  o--o  o
        // 
        // o  o--o  o--o  o
        num_constraint_groups = ((width-2)/2) * ((height-2)/2) + (width-2) +   
                                (height-2);
        num_blocks = UP_DIV(num_constraint_groups, threads_per_block);
        satisfy_case4_kernel<<<num_blocks, threads_per_block>>>(dptr);
        cudaDeviceSynchronize();
    }
}

void cuda_cloth::simulate_timestep()
{
    /*static int num = 0;
    printf("iter %d \n", num++);
    for(int i = 0; i < num_particles; i++)
    {
        printf("CUDA Pos: %d ==> (%f, %f, %f)\n", i, host_pos_array[i].x,
                                host_pos_array[i].y, host_pos_array[i].z);
    }*/
    float3 *dptr, *nptr;
    size_t num_bytes = get_cuda_device_ptr(&cuda_pos_vbo_resource, &dptr);
    size_t norm_bytes = get_cuda_device_ptr(&cuda_normal_vbo_resource, &nptr);
    double forces_start = CycleTimer::currentSeconds();
    apply_forces(dptr, nptr);
    double forces_end = CycleTimer::currentSeconds();

    double update_start = CycleTimer::currentSeconds();
    update_positions(dptr);
    double update_end = CycleTimer::currentSeconds();

    double constraint_start = CycleTimer::currentSeconds();
    satisfy_constraints(dptr);
    double constraint_end = CycleTimer::currentSeconds();

    double transfer_start = CycleTimer::currentSeconds();
    
    double transfer_end = CycleTimer::currentSeconds();

    printf("----------------------------------------\n");
    printf("Apply Forces : %.3f ms \n", forces_end-forces_start);
    printf("Update Positions : %.3f ms \n", update_end-update_start);
    printf("Satisfy Constraints : %.3f ms \n", constraint_end-constraint_start);
    printf("Transfer to CPU : %.3f ms \n", transfer_end-transfer_start);
    printf("----------------------------------------\n");
}
