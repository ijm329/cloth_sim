#include "cuda_cloth.h"

CudaCloth::CudaCloth(int n)
{
    num_particles_width = n;
    num_particles_height = n;
    particles = (particle *)malloc(sizeof(particle) * num_particles);
    if((particles == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    cudaMalloc(&dev_particles, sizeof(particle) * num_particles);
}

CudaCloth::CudaCloth(int w, int h)
{
    num_particles_width = n;
    num_particles_height = n;
    particles = (particle *)malloc(sizeof(particle) * num_particles);
    if((particles == NULL))
    {
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    cudaError_t rc = cudaMalloc(&dev_particles, sizeof(particle) * num_particles);
    if (rc != cudaSuccess)
    {
        std::cout << "GPU Allocation Failure" << std::endl;
        exit(1);
    }
}

// initializes cloth positions and allocates memory region in the device and 
// copies the particle buffer into that region
void Cloth::init()
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
    //copy initialized data to device
    cudaError_t rc = cudaMemcpy(dev_particles, particles, sizeof(particle) * num_particles, 
                                cudaMemcpyHostToDevice);
    if(rc != cudaSuccess)
    {
        std::cout << "GPU Transfer Failure" << std::endl;
        exit(1);
    }
}

__device__ void load_particles(particle **blk_particles)
{
    //row in col within the entire grid of threads
    int row = blockIdx.x * blockDim.x + threadId.x;
    int col = blockIdx.y * blockDim.y + threadId.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x = blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    vector3D tot_force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    //cooperatively load into the array the necessary particles starting 
    //with those contained in the thread block.
    int idx = row * num_particles_width + col;
    blk_particles[row + 1][col + 1] = dev_particles[idx];
    //for border particles, load the immediate top, left, bottom, and right
    //load tops so only if you're in the top row of your block
    if(threadId.y == 0)
    {
        //ensure you're not the top row in the system
        if(row != 0)
        {
            int top = (row - 1) * num_particles_width + col;
            blk_particles[0][thread.x + 1] = dev_particles[top];
        }
        else
        {
            blk_particles[0][thread.x + 1] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //bottom particles
    else if(thread.y == blockDim.y - 1)
    {
        //not the bottom row in the system
        if(row != num_particles_height - 1)
        {
            int btm = (row + 1) * num_particles_width + col;
            blk_particles[thread.y + 2][thread.x + 1].pos = dev_particles[btm];
        }
        else
        {
            blk_particles[thread.y + 2][thread.x + 1].pos = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //left
    if(thread.x == 0)
    {
        //if you're not the left edge in the system 
        if(col != 0)
        {
            int left = idx - 1;
            blk_particles[threadId.y + 1][0] = dev_particles[left];
        }
        else
        {
            blk_particles[threadId.y + 1][0] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //right 
    else if(thread.x == blockDim.x - 1)
    {
        //if you're not the right edge in the system
        if(col != num_particles_width - 1)
        {
            int right = idx + 1;
            blk_particles[threadId.y + 1][thread.x + 2] = dev_particles[right];
        }
        else
        {
            blk_particles[threadId.y + 1][thread.x + 2] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //corners 
    //top left
    if(thread.x == 0 && thread.y == 0)
    {
        //if not at the top of the system and not at the top left edge
        if(row != 0 && col != 0)
        {
            int top_left = (row - 1) * num_particles_width + (col - 1);
            blk_particles[0][0] = dev_particles[top_left];
        }
        else
        {
            blk_particles[0][0] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //top right
    else if(thread.x == blockDim.x - 1 && thread.y == 0)
    {
        if(row != 0 && col != num_particles_width - 1)
        {
            int top_right = (row - 1) * num_particles_width + (col + 1);
            blk_particles[0][shared_mem_x - 1] = dev_particles[top_right];
        }
        else
        {
            blk_particles[0][shared_mem_x - 1] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //bottom left
    else if(thread.x == 0 && thread.y == blockDim.y - 1)
    {
        if(row != num_particles_height - 1 && col != 0)
        {
            int btm_left = (row + 1) * num_particles_width + col - 1;
            blk_particles[shared_mem_y - 1][0] = dev_particles[btm_left];
        }
        else
        {
            blk_particles[shared_mem_y - 1][0] = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //bottom right 
    else if(thread.x == blockDim.x - 1 && thread.y == blockDim.y - 1)
    {
        if(row != num_particles_height - 1 && col != num_particles_width - 1)
        {
            int btm_right = (row + 1) * num_particles_width + col + 1;
            blk_particles[shared_mem_y - 1][shared_mem_x - 1] = dev_particles[btm_right];
        }
    }
}

__global__ void apply_all_forces()
{
    //row in col within the entire grid of threads
    int row = blockIdx.x * blockDim.x + threadId.x;
    int col = blockIdx.y * blockDim.y + threadId.y;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x = blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    vector3D force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f)
    __shared__ particle blk_particles[shared_mem_y][shared_mem_x];
    if(row < num_particles_height && col < num_particles_width)
    {
        load_particles(blk_particles);
    }
    __syncthreads();
}

void CudaCloth::apply_forces()
{
  //setup invocaiton for application of all forces
}

__global__ void update_all_positions()
{
    int row = blockIdx.x * blockDim.x + threadId.x;
    int col = blockIdx.y * blockDim.y + threadId.y;
    if(row < num_particles_height && col < num_particles_width)
    {
        int i = row * num_particles_width + col;
        vector3D temp(dev_particles[i].pos);
        vector3D acc = particles[i].force/PARTICLE_MASS;
        particles[i].pos += (particles[i].pos - particles[i].prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        particles[i].prev_pos = temp;
    }
}

void CudaCloth::update_positions()
{
    //setup invocation for update positions

}

void CudaCloth::simulate_timestep()
{
    apply_forces();
    update_positions();
    satisfy_constraints();
}

