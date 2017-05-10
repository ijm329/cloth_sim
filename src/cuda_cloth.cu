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

__device__ void load_particles(int width, int height, particle *dev_parts, particle **blk_particles)
{
    //row in col within the entire grid of threads
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x = blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    vector3D tot_force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    //cooperatively load into the array the necessary particles starting 
    //with those contained in the thread block.
    int idx = row * width + col;
    blk_particles[row + 1][col + 1] = dev_parts[idx];
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
            blk_particles[0][threadIdx.x + 1].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[threadIdx.y + 2][threadIdx.x + 1].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[threadIdx.y + 1][0].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[threadIdx.y + 1][threadIdx.x + 2].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[0][0].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[0][shared_mem_x - 1].pos = vector3D(-2.0, -2.0, -2.0);
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
            blk_particles[shared_mem_y - 1][0].pos = vector3D(-2.0, -2.0, -2.0);
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

__global__ void apply_all_forces(int width, int height, particle *dev_parts)
{
    //row in col within the entire grid of threads
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x =  blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    vector3D force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    __shared__ particle blk_particles[TPB_Y + 2][TPB_X + 2];
    if(row < height && col < width)
    {
        load_particles(width, height, dev_parts, (particle **)blk_particles);
    }
    __syncthreads();
}

void CudaCloth::apply_forces()
{
  printf("Hi good morning\n");
  //setup invocaiton for application of all forces
}

__global__ void update_all_positions(int width, int height, particle *dev_parts)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < height && col < width)
    {
        int i = row * width + col;
        particle curr = dev_parts[i];
        vector3D temp(curr.pos);
        vector3D acc = curr.force/PARTICLE_MASS;
        curr.pos += (curr.pos - curr.prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        curr.prev_pos = temp;
        dev_parts[i] = curr;
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
    //apply_forces();
    update_positions();
    //satisfy_constraints();
}

