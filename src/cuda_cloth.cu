#include "cuda_cloth.h"

particle *dev_particles;

CudaCloth::CudaCloth(int n)
{
    num_particles_width = n;
    num_particles_height = n;
    num_particles = num_particles_width * num_particles_height;
    particles = (particle *)malloc(sizeof(particle) * num_particles);
    if((particles == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    GPU_ERR_CHK(cudaMalloc(&dev_particles, sizeof(particle) * num_particles));
}

CudaCloth::CudaCloth(int w, int h)
{
    num_particles_width = w;
    num_particles_height = h;
    particles = (particle *)malloc(sizeof(particle) * num_particles);
    if((particles == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    GPU_ERR_CHK(cudaMalloc(&dev_particles, sizeof(particle) * num_particles)); 
}

CudaCloth::~CudaCloth()
{
    free(particles);
    cudaFree(dev_particles);
}

// Get particles back from the device for rendering 
void CudaCloth::get_particles()
{
    GPU_ERR_CHK(cudaMemcpy(particles, dev_particles, sizeof(particle) * num_particles, 
                                cudaMemcpyDeviceToHost));
    /*for(int i = 0; i < num_particles; i++)
    {
      printf("%d {%f, %f, %f}\n", i, particles[i].pos.x, particles[i].pos.y, particles[i].pos.z);
    }*/
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

            particles[i*num_particles_width + j].pos = {x, 1.0f, z};
            particles[i*num_particles_width + j].prev_pos = {x, 1.0f, z};
            particles[i*num_particles_width + j].force = {0.0f, 0.0f, 0.0f};
        }
    }
    //copy initialized data to device
    GPU_ERR_CHK(cudaMemcpy(dev_particles, particles, sizeof(particle) * num_particles, 
                                cudaMemcpyHostToDevice));
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

__device__ inline float3 get_velocity(particle p)
{
    return (p.pos - p.prev_pos) * (1.0f / TIME_STEP);
}

__device__ inline float get_spr_const(spring_type_t type)
{
    if(type == STRUCTURAL)
        return K_STRUCT;
    
    else if(type == SHEAR)
        return K_SHEAR;
    else 
        return K_FLEXION;
}

__device__ inline float get_spr_damp(spring_type_t type)
{
    if(type == STRUCTURAL)
        return DAMPING_STRUCT;
    
    else if(type == SHEAR)
        return DAMPING_SHEAR;
    else 
        return DAMPING_FLEXION;
}

__device__ inline float get_spr_len(int width, int height, spring_type_t type)
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


__device__ float3 compute_spring_force(particle p1, particle p2, float len, spring_type_t type)
{
    float3 dir = p2.pos - p1.pos;
    float3 rest = len * (normalize(dir));
    float spr_const = get_spr_const(type);
    float damp_const = get_spr_damp(type);
    float3 disp = (spr_const * dir) - (spr_const * rest);
    float3 vel = get_velocity(p2) - get_velocity(p1);
    float3 force = -disp - (damp_const * vel);
    return force;
}

__global__ void apply_all_forces(int width, int height, particle *dev_parts)
{
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x =  blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    float3 tot_force = make_float3(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    __shared__ particle blk_particles[TPB_Y + 2][TPB_X + 2];
    //row 1: up, row 2: down , row 3: left row 4: right
    __shared__ particle flexion_parts[4][TPB_X];
    if(row < height && col < width)
    {
        //cooperatively load into the array the necessary particles starting 
        //with those contained in the thread block.
        int idx = (row * width) + col;
        blk_particles[threadIdx.y + 1][threadIdx.x + 1] = dev_parts[idx];
        //for border particles, load the immediate top, left, bottom, and right
        //load tops so only if you're in the top row of your block
        if(threadIdx.y == 0)
        {
            //ensure you're not the top row in the system
            if(row != 0)
            {
                int top = (row - 1) * width + col;
                blk_particles[0][threadIdx.x + 1] = dev_parts[top];
            }
            else
            {
                blk_particles[0][threadIdx.x + 1].pos = make_float3(-2.0, -2.0, -2.0);
            }
            //ensure that you have an upper flexion spring 
            if(row - 2 >= 0)
            {
                int f_top = (row - 2) * width + col;
                flexion_parts[0][threadIdx.x] = dev_parts[f_top];
            }
            else
            {
                flexion_parts[0][threadIdx.x].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //bottom particles
        if(threadIdx.y == blockDim.y - 1)
        {
            //not the bottom row in the system
            if(row != height - 1)
            {
                int btm = (row + 1) * width + col;
                blk_particles[threadIdx.y + 2][threadIdx.x + 1] = dev_parts[btm];
            }
            else
            {
                blk_particles[threadIdx.y + 2][threadIdx.x + 1].pos = make_float3(-2.0, -2.0, -2.0);
            }
            if(row + 2 < height)
            {
                int f_btm = (row + 2) * width + col;
                flexion_parts[1][threadIdx.x] = dev_parts[f_btm];
            }
            else
            {
                flexion_parts[1][threadIdx.x].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //left
        if(threadIdx.x == 0)
        {
            //if you're not the left edge in the system 
            if(col != 0)
            {
                int left = idx - 1;
                blk_particles[threadIdx.y + 1][0] = dev_parts[left];
            }
            else
            {
                blk_particles[threadIdx.y + 1][0].pos = make_float3(-2.0, -2.0, -2.0);
            }
            if(col - 2 >= 0)
            {
                int f_left = idx - 2; 
                flexion_parts[2][threadIdx.y] = dev_parts[f_left];
            }
            else
            {
                flexion_parts[2][threadIdx.y].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //right 
        if(threadIdx.x == blockDim.x - 1)
        {
            //if you're not the right edge in the system
            if(col != width - 1)
            {
                int right = idx + 1;
                blk_particles[threadIdx.y + 1][threadIdx.x + 2] = dev_parts[right];
            }
            else
            {
                blk_particles[threadIdx.y + 1][threadIdx.x + 2].pos = make_float3(-2.0, -2.0, -2.0);
            }
            if(col + 2 < width)
            {
                int f_right = idx + 2;
                flexion_parts[3][threadIdx.y] = dev_parts[f_right];
            }
            else
            {
                flexion_parts[3][threadIdx.y].pos = make_float3(-2, -2, -2);
            }
        }
        //corners 
        //top left
        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            //if not at the top of the system and not at the top left edge
            if(row != 0 && col != 0)
            {
                int top_left = (row - 1) * width + (col - 1);
                blk_particles[0][0] = dev_parts[top_left];
            }
            else
            {
                blk_particles[0][0].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //top right
        if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
        {
            if(row != 0 && col != width - 1)
            {
                int top_right = (row - 1) * width + (col + 1);
                blk_particles[0][shared_mem_x - 1] = dev_parts[top_right];
            }
            else
            {
                blk_particles[0][shared_mem_x - 1].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //bottom left
        if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != 0)
            {
                int btm_left = (row + 1) * width + (col - 1);
                blk_particles[shared_mem_y - 1][0] = dev_parts[btm_left];
            }
            else
            {
                blk_particles[shared_mem_y - 1][0].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //bottom right 
        if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != width - 1)
            {
                int btm_right = (row + 1) * width + (col + 1);
                blk_particles[shared_mem_y - 1][shared_mem_x - 1] = dev_parts[btm_right];
            }
        }
    }
    __syncthreads();
    //Now that all necessary particles have been loaded, compute wind forces
    //computing wind forces and spring forces
    if(row < height && col < width)
    {
        //your row and col within the shared mem matrix
        int sblk_row = threadIdx.y + 1;
        int sblk_col = threadIdx.x + 1;
        int struct_len = get_spr_len(width, height, STRUCTURAL);
        int shear_len = get_spr_len(width, height, SHEAR);
        int flex_len = 2 * struct_len;
        particle curr = blk_particles[sblk_row][sblk_col];
        int idx = row * width + col;
        if(row != 0 && col != width - 1)
        {
            particle top = blk_particles[sblk_row - 1][sblk_col];
            particle top_right = blk_particles[sblk_row - 1][sblk_col + 1];
            tot_force += compute_wind_force(top_right.pos, top.pos, curr.pos);
            particle right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, top_right.pos, curr.pos);
            //tot_force += compute_spring_force(curr, top_right, shear_len, SHEAR);
        }
        if(row != height - 1 && col != width - 1)
        {
            particle bottom = blk_particles[sblk_row + 1][sblk_col];
            particle right = blk_particles[sblk_row][sblk_col + 1];
            tot_force += compute_wind_force(right.pos, curr.pos, bottom.pos);
            particle bottom_right = blk_particles[sblk_row + 1][sblk_col + 1];
            //tot_force += compute_spring_force(curr, bottom_right, shear_len, SHEAR);
        }
        if(row != height - 1 && col != 0)
        {
            particle bottom = blk_particles[sblk_row + 1][sblk_col];
            particle bottom_left = blk_particles[sblk_row + 1][sblk_col - 1];
            tot_force += compute_wind_force(bottom.pos, curr.pos, bottom_left.pos);
            particle left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, left.pos, bottom_left.pos);
            //tot_force += compute_spring_force(curr, bottom_left, shear_len, SHEAR);
        }
        if(row != 0 && col != 0)
        {
            particle upper = blk_particles[sblk_row - 1][sblk_col];
            particle left = blk_particles[sblk_row][sblk_col - 1];
            tot_force += compute_wind_force(curr.pos, upper.pos, left.pos);
            particle upper_left = blk_particles[sblk_row - 1][sblk_col - 1];
            //tot_force += compute_spring_force(curr, upper_left, shear_len, SHEAR);
        }
        //structural forces
        /*if(col != 0)
            tot_force += compute_spring_force(curr, blk_particles[sblk_row][sblk_col - 1], 
                                              struct_len, STRUCTURAL);
        if(col != width - 1)
            tot_force += compute_spring_force(curr, blk_particles[sblk_row][sblk_col + 1], 
                                              struct_len, STRUCTURAL);
        if(row != 0)
            tot_force += compute_spring_force(curr, blk_particles[sblk_row - 1][sblk_col], 
                                              struct_len, STRUCTURAL);
        if(row != height - 1)
            tot_force += compute_spring_force(curr, blk_particles[sblk_row + 1][sblk_col], 
                                              struct_len, STRUCTURAL);

        //flexion
        //starting with upper springs 
        if(row - 2 >= 0)
        {
            //check to see which array it's in 
            particle f_top;
            if(sblk_row - 2 >= 0)
            {
                f_top = blk_particles[sblk_row - 2][sblk_col];
            }
            else
            {
                f_top = flexion_parts[0][threadIdx.x];
            }
            tot_force += compute_spring_force(curr, f_top, flex_len, FLEXION);
        }
        if(row + 2 < height)
        {
            particle f_btm;
            if(sblk_row + 2 < shared_mem_y)
            {
                f_btm = blk_particles[sblk_row + 2][sblk_col];
            }
            else
            {
                f_btm = flexion_parts[1][threadIdx.x];
            }
            tot_force += compute_spring_force(curr, f_btm, flex_len, FLEXION);
        }
        if(col - 2 >= 0)
        {
            particle f_left;
            if(sblk_col - 2 >= 0)
            {
                f_left = blk_particles[sblk_row][sblk_col - 2];
            }
            else
            {
                f_left = flexion_parts[2][threadIdx.y];
            }
            tot_force += compute_spring_force(curr, f_left, flex_len, FLEXION);
        }
        if(col + 2 < width)
        {
            particle f_right;
            if(sblk_col + 2 < shared_mem_x)
            {
                f_right = blk_particles[sblk_row][sblk_col + 2];
            }
            else
            {
                f_right = flexion_parts[3][threadIdx.y];
            }
            tot_force += compute_spring_force(curr, f_right, flex_len, FLEXION);
        }*/
        dev_parts[idx].force = tot_force;
    }
}

void CudaCloth::apply_forces()
{
  //setup invocaiton for application of all forces
  dim3 threadsPerBlock(TPB_X, TPB_Y,1);
  dim3 numBlocks(num_particles_width / TPB_X, num_particles_height / TPB_Y, 1);
  apply_all_forces<<<numBlocks, threadsPerBlock>>>(num_particles_width, 
                                                         num_particles_height, 
                                                         dev_particles);
  cudaDeviceSynchronize();
}

__global__ void update_all_positions(int width, int height, particle *dev_parts)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < height && col < width)
    {
        int i = row * width + col;
        particle curr = dev_parts[i];
        float3 temp = make_float3(curr.pos.x, curr.pos.y, curr.pos.z);
        float3 acc = curr.force/PARTICLE_MASS;
        curr.pos += (curr.pos - curr.prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        curr.prev_pos = temp;
        dev_parts[i].pos = curr.pos;
        dev_parts[i].prev_pos = curr.prev_pos;
    }
}

void CudaCloth::update_positions()
{
    dim3 threadsPerBlock(TPB_X, TPB_Y, 1);
    dim3 numBlocks(num_particles_width / TPB_X, num_particles_height / TPB_Y, 1);
    update_all_positions<<<numBlocks, threadsPerBlock>>>(num_particles_width, 
                                                         num_particles_height, 
                                                         dev_particles);
    cudaDeviceSynchronize();
}

void CudaCloth::simulate_timestep()
{

    apply_forces();
    update_positions();
    //satisfy_constraints();
}

