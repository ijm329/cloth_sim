#include "cuda_cloth.h"

#define POS_INDEX 0
#define PREV_POS_INDEX 1

struct cloth_constants
{
    int num_particles_width;
    int num_particles_height;
    float struct_spring_len;
    float shear_spring_len;
    float3 *pos_array;
    float3 *prev_pos_array;
    float3 *force_array;
};

typedef struct
{
    float3 pos;
    float3 prev_pos;
} particle_pos_data_t;

__constant__ cloth_constants cuda_cloth_params;

CudaCloth::CudaCloth(int n)
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
    GPU_ERR_CHK(cudaMalloc(&dev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_prev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_force_array, sizeof(float3) * num_particles));
}

CudaCloth::CudaCloth(int w, int h)
{
    num_particles_width = w;
    num_particles_height = h;
    host_pos_array = (float3 *)malloc(sizeof(float3) * num_particles);
    if((host_pos_array == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    GPU_ERR_CHK(cudaMalloc(&dev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_prev_pos_array, sizeof(float3) * num_particles));
    GPU_ERR_CHK(cudaMalloc(&dev_force_array, sizeof(float3) * num_particles));
}

CudaCloth::~CudaCloth()
{
    free(host_pos_array);
    cudaFree(dev_pos_array);
    cudaFree(dev_prev_pos_array);
    cudaFree(dev_force_array);
}

// Get particles back from the device for rendering 
void CudaCloth::get_particles()
{
    GPU_ERR_CHK(cudaMemcpy(host_pos_array, dev_pos_array,
                sizeof(float3) * num_particles, cudaMemcpyDeviceToHost));
    /*
     * for(int i = 0; i < num_particles; i++)
     * {
     *     printf("%d {%f, %f, %f}\n", i, particles[i].pos.x, particles[i].pos.y,
     *                                 particles[i].pos.z);
     * }
     * */
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

// initializes cloth positions and allocates memory region in the device and 
// copies the particle buffer into that region
void CudaCloth::init()
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
    //copy initialized data to device
    GPU_ERR_CHK(cudaMemcpy(dev_pos_array, host_pos_array,
                sizeof(float3) * num_particles, cudaMemcpyHostToDevice));
    GPU_ERR_CHK(cudaMemcpy(dev_prev_pos_array, dev_pos_array,
                sizeof(float3) * num_particles, cudaMemcpyDeviceToDevice));
    GPU_ERR_CHK(cudaMemset(dev_force_array, 0, sizeof(float3) * num_particles));

    //set up cloth simulation parameters in read only memory
    cloth_constants params;
    params.num_particles_width = num_particles_width;
    params.num_particles_height = num_particles_height;
    params.pos_array = dev_pos_array;
    params.prev_pos_array = dev_prev_pos_array;
    params.force_array = dev_force_array;
    params.struct_spring_len = get_spring_len(num_particles_width,
                               num_particles_height, STRUCTURAL);
    params.shear_spring_len = get_spring_len(num_particles_width,
                              num_particles_height, SHEAR);
    GPU_ERR_CHK(cudaMemcpyToSymbol(cuda_cloth_params, &params,
                                  sizeof(cloth_constants)));
}

void CudaCloth::render(float rotate_x, float rotate_y, float translate_z)
{
    //transform the cloth's position in the world space based on 
    //camera parameters
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    render_particles();
}

void CudaCloth::render_particles()
{
    glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    glColor3f(0.0, 1.0, 1.0);
    glVertexPointer(3, GL_FLOAT, sizeof(float3), &(host_pos_array[0].x));
    //glColorPointer(3, GL_FLOAT, sizeof(float3), &(particles[0].color.x));
    glDrawArrays(GL_POINTS, 0, num_particles);
    glDisableClientState(GL_VERTEX_ARRAY);
    //glDisableClientState(GL_COLOR_ARRAY);
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

__device__ float3 compute_wind_force(float3 p1, float3 p2, float3 p3)
{
    float3 line1 = p2 - p3;
    float3 line2 = p3 - p1;

    //normal to the triangle
    float3 norm = cross(line1, line2);
    float3 wind = make_float3(WIND_X, WIND_Y, WIND_Z);
    float3 wind_force = norm * (dot(normalize(norm), wind));
    return wind_force; 
}

__device__ float3 compute_spring_force(float3 p2_pos, float3 p1_pos,
                                       float3 p2_prev_pos, 
                                       float3 p1_prev_pos, 
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
    return force;
}

__device__ __inline__ void load_particle_pos_data(
        particle_pos_data_t blk_particles[][TPB_X+2],
        float3 *f_top[2],
        float3 *f_btm[2],
        float3 *f_left[2],
        float3 *f_right[2])
{
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    float3 *dev_pos_array = cuda_cloth_params.pos_array;
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
                blk_particles[0][sblk_col].prev_pos = dev_prev_pos_array[index];
            }
            else
                blk_particles[0][sblk_col].pos = make_float3(MIN_BOUND-1,
                                                MIN_BOUND-1, MIN_BOUND-1);
        }

        //bottom particles
        if(threadIdx.y == blockDim.y - 1)
        {
            //not the bottom row in the system
            if(row != height - 1)
            {
                int index = (row + 1) * width + col;
                blk_particles[sblk_row + 1][sblk_col].pos = dev_pos_array[index];
                blk_particles[sblk_row + 1][sblk_col].prev_pos =
                                                        dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row + 1][sblk_col].pos = make_float3(
                                    MIN_BOUND-1, MIN_BOUND-1, MIN_BOUND-1);
        }

        //left particles
        if(threadIdx.x == 0)
        {
            //if you're not the left edge in the system 
            if(col != 0)
            {
                int index = idx-1;
                blk_particles[sblk_row][0].pos = dev_pos_array[index];
                blk_particles[sblk_row][0].prev_pos = dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row][0].pos = make_float3(MIN_BOUND-1,
                                                MIN_BOUND-1, MIN_BOUND-1);
        }

        //right particles
        if(threadIdx.x == blockDim.x - 1)
        {
            //if you're not the right edge in the system
            if(col != width - 1)
            {
                int index = idx+1;
                blk_particles[sblk_row][sblk_col + 1].pos = dev_pos_array[index];
                blk_particles[sblk_row][sblk_col + 1].prev_pos =
                                                        dev_prev_pos_array[index];
            }
            else
                blk_particles[sblk_row][sblk_col + 1].pos = make_float3(
                                    MIN_BOUND-1, MIN_BOUND-1, MIN_BOUND-1);
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
                blk_particles[0][0].pos = make_float3(MIN_BOUND-1, MIN_BOUND-1,
                                                MIN_BOUND-1);
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
                blk_particles[0][shared_mem_x - 1].pos = make_float3(MIN_BOUND-1,
                                                        MIN_BOUND-1, MIN_BOUND-1);
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
                blk_particles[shared_mem_y - 1][0].pos = make_float3(MIN_BOUND-1,
                                                        MIN_BOUND-1, MIN_BOUND-1);
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
                            make_float3(MIN_BOUND-1, MIN_BOUND-1, MIN_BOUND-1);
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
        float3 *f_right[2])
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

    //Now that all necessary particles have been loaded, compute wind forces
    //computing wind forces and spring forces
    if(row < height && col < width)
    {
        //your row and col within the shared mem matrix
        particle_pos_data_t curr = blk_particles[sblk_row][sblk_col];
        if(row != 0 && col != width - 1)
        {
            particle_pos_data_t top = blk_particles[sblk_row - 1][sblk_col];
            particle_pos_data_t top_right = 
                blk_particles[sblk_row-1][sblk_col+1];
            tot_force += compute_wind_force(top_right.pos, top.pos, curr.pos);
            particle_pos_data_t right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, top_right.pos, curr.pos);
            tot_force += compute_spring_force(curr.pos, top_right.pos, 
                         curr.prev_pos, top_right.prev_pos, shear_len, SHEAR);
        }
        if(row != height - 1 && col != width - 1)
        {
            particle_pos_data_t bottom = blk_particles[sblk_row + 1][sblk_col];
            particle_pos_data_t right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, curr.pos, bottom.pos);
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
                                            bottom_left.pos);
            particle_pos_data_t left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, left.pos, 
                                            bottom_left.pos);
            tot_force += compute_spring_force(curr.pos, bottom_left.pos, 
                        curr.prev_pos, bottom_left.prev_pos, shear_len, SHEAR);
        }
        if(row != 0 && col != 0)
        {
            particle_pos_data_t top = blk_particles[sblk_row - 1][sblk_col];
            particle_pos_data_t left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, top.pos, left.pos);
            particle_pos_data_t top_left =
                        blk_particles[sblk_row - 1][sblk_col - 1];
            tot_force += compute_spring_force(curr.pos, top_left.pos, 
                        curr.prev_pos, top_left.prev_pos, shear_len, SHEAR);
        }

        //structural forces
        if(col != 0)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row][sblk_col - 1].pos, curr.prev_pos,
              blk_particles[sblk_row][sblk_col - 1].prev_pos, struct_len, STRUCTURAL);
        if(col != width - 1)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row][sblk_col + 1].pos, curr.prev_pos,
              blk_particles[sblk_row][sblk_col + 1].prev_pos, struct_len, STRUCTURAL);
        if(row != 0)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row - 1][sblk_col].pos, curr.prev_pos,
              blk_particles[sblk_row - 1][sblk_col].prev_pos, struct_len, STRUCTURAL);
        if(row != height - 1)
            tot_force += compute_spring_force(curr.pos,
              blk_particles[sblk_row + 1][sblk_col].pos, curr.prev_pos,
              blk_particles[sblk_row + 1][sblk_col].prev_pos, struct_len, STRUCTURAL);

        //flexion forces 
        /*if(f_top[POS_INDEX]) 
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

__global__ void apply_forces_kernel()
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

    load_particle_pos_data(blk_particles, f_top, f_btm, f_left, f_right);
    __syncthreads();
    float3 tot_force = compute_particle_forces(blk_particles, f_top, f_btm,
                                               f_left, f_right);
    if((row < height) && (col < width))
      dev_force_array[idx] = tot_force;
}

void CudaCloth::apply_forces()
{
  //setup invocaiton for application of all forces
  dim3 threadsPerBlock(TPB_X, TPB_Y, 1);
  dim3 numBlocks(UP_DIV(num_particles_width, TPB_X), 
                 UP_DIV(num_particles_height, TPB_Y), 1);
  apply_forces_kernel<<<numBlocks, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}

__global__ void update_positions_kernel()
{
    int width = cuda_cloth_params.num_particles_width;
    int height = cuda_cloth_params.num_particles_height;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //if valid particle index
    if(row < height && col < width)
    {
        int i = row * width + col;
     
        float3 curr_pos = cuda_cloth_params.pos_array[i];
        float3 curr_prev_pos = cuda_cloth_params.prev_pos_array[i];
        float3 curr_force = cuda_cloth_params.force_array[i];
        float3 temp = make_float3(curr_pos.x, curr_pos.y, curr_pos.z);
        float3 acc = curr_force/PARTICLE_MASS;
        curr_pos += (curr_pos - curr_prev_pos + acc * TIME_STEP * TIME_STEP);
        curr_prev_pos = temp;
        cuda_cloth_params.pos_array[i] = curr_pos;
        cuda_cloth_params.prev_pos_array[i] = curr_prev_pos;
        
        //fixed particles
        if((row == 0 && col == 0) || (row == 0 && col == width - 1))
        {
          float x = (float)col/width;
          float  z = (float)row/height;
          x = BOUND_LENGTH*x + MIN_BOUND;
          z = BOUND_LENGTH*z + MIN_BOUND;
          cuda_cloth_params.pos_array[i] = make_float3(x,1.0f,z);
        }
    }
}

void CudaCloth::update_positions()
{
    dim3 threadsPerBlock(TPB_X, TPB_Y, 1);
    dim3 numBlocks(UP_DIV(num_particles_width, TPB_X), 
                   UP_DIV(num_particles_height, TPB_Y), 1);
    update_positions_kernel<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}

void CudaCloth::satisfy_constraints()
{
}

void CudaCloth::simulate_timestep()
{
    /*static int num = 0;
    printf("iter %d \n", num++);
    for(int i = 0; i < num_particles; i++)
    {
        printf("CUDA Pos: %d ==> (%f, %f, %f)\n", i, host_pos_array[i].x,
                                host_pos_array[i].y, host_pos_array[i].z);
    }*/
    apply_forces();
    update_positions();
    satisfy_constraints();
    get_particles();
}
