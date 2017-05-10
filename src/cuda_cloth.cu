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

__global__ void apply_all_forces(int width, int height, particle *dev_parts)
{
    //row in col within the entire grid of threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x =  blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    float3 force = make_float3(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    __shared__ particle blk_particles[TPB_Y + 2][TPB_X + 2];
    if(row < height && col < width)
    {
        float3 tot_force = make_float3(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
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
        }
        //bottom particles
        else if(threadIdx.y == blockDim.y - 1)
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
        }
        //right 
        else if(threadIdx.x == blockDim.x - 1)
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
        else if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
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
        else if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != 0)
            {
                int btm_left = (row + 1) * width + col - 1;
                blk_particles[shared_mem_y - 1][0] = dev_parts[btm_left];
            }
            else
            {
                blk_particles[shared_mem_y - 1][0].pos = make_float3(-2.0, -2.0, -2.0);
            }
        }
        //bottom right 
        else if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
        {
            if(row != height - 1 && col != width - 1)
            {
                int btm_right = (row + 1) * width + col + 1;
                blk_particles[shared_mem_y - 1][shared_mem_x - 1] = dev_parts[btm_right];
            }
        }
    }
    __syncthreads();
    //Now that all necessary particles have been loaded, compute wind forces
    //computing wind force 1 
    if(row < height && col < width)
    {
        //your row and col within the shared mem matrix
        int sblk_row = threadIdx.y + 1;
        int sblk_col = threadIdx.x + 1;
        particle curr = blk_particles[sblk_row][sblk_col];
        if(row != 0 && col != width - 1)
        {
            float3 top = blk_particles[sblk_row - 1][sblk_col].pos;
            float3 top_right = blk_particles[sblk_row - 1][sblk_col + 1].pos;
            curr.force += compute_wind_force(top_right, top, curr.pos);
            float3 right = blk_particles[sblk_row][sblk_col + 1].pos;
            curr.force += compute_wind_force(right, top_right, curr.pos);
        }
        if(row != height && col != width - 1)
        {
            float3 bottom = blk_particles[sblk_row + 1][sblk_col].pos;
            float3 right = blk_particles[sblk_row][sblk_col + 1].pos;
            curr.force += compute_wind_force(right, curr.pos, bottom);
        }
        if(row != height && col != 0)
        {
            float3 bottom = blk_particles[sblk_row + 1][sblk_col].pos;
            float3 bottom_left = blk_particles[sblk_row + 1][sblk_col - 1].pos;
            curr.force += compute_wind_force(bottom, curr.pos, bottom_left);
            float3 left = blk_particles[sblk_row][sblk_col - 1].pos;
            curr.force += compute_wind_force(curr.pos, left, bottom_left);
        }
        if(row != 0 && col != 0)
        {
            float3 upper = blk_particles[sblk_row - 1][sblk_col].pos;
            float3 left = blk_particles[sblk_row][sblk_col - 1].pos;
            curr.force += compute_wind_force(curr.pos, upper, left);
        }
        //write back forces for now
        dev_parts[row * width + col].force = curr.force;
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < height && col < width)
    {
        int i = row * width + col;
        /*particle curr = dev_parts[i];
        float3 temp(curr.pos);
        float3 acc = curr.force/PARTICLE_MASS;
        curr.pos += (curr.pos - curr.prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        curr.prev_pos = temp;*/
        dev_parts[i].pos = make_float3(-9.0,-9.0,-9.0);
    }
}

void CudaCloth::update_positions()
{
    printf("Yes hello\n");
    //setup invocation for update positions
    dim3 threadsPerBlock(TPB_X, TPB_Y,1);
    dim3 numBlocks(num_particles_width / TPB_X, num_particles_height / TPB_Y, 1);
    update_all_positions<<<numBlocks, threadsPerBlock>>>(num_particles_width, 
                                                         num_particles_height, 
                                                         dev_particles);
}

void CudaCloth::simulate_timestep()
{
    apply_forces();
    //update_positions();
    //satisfy_constraints();
}

