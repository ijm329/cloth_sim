#include "cuda_cloth.h"

__constant__ GlobalConstants cuConstRendererParams;

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
    cudaError_t rc = cudaMalloc(&dev_particles, sizeof(particle) * num_particles);
    if(rc != cudaSuccess)
    {
        std::cout << "GPU Allocation Failure" << std::endl;
        exit(1);
    }
    GlobalConstants params; 
    params.num_particles_width = num_particles_width;
    params.num_particles_height = num_particles_height;
    params.dev_particles = dev_particles;
    rc = cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants), 
                                        0, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess)
    {
        std::cout << "GPU Symbol Transfer Failure" << std::endl;
        exit(1);
    }
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
    cudaError_t rc = cudaMalloc(&dev_particles, sizeof(particle) * num_particles);
    if (rc != cudaSuccess)
    {
        std::cout << "GPU Allocation Failure" << std::endl;
        exit(1);
    }
    GlobalConstants params;
    params.num_particles_width = num_particles_width;
    params.num_particles_height = num_particles_height;
    params.dev_particles = dev_particles;
    rc = cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants), 
                                        0, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess)
    {
        std::cout << "GPU Symbol Transfer Failure" << std::endl;
        exit(1);
    }
}

CudaCloth::~CudaCloth()
{
    free(particles);
    cudaFree(dev_particles);
}

// Get particles back from the device for rendering 
void CudaCloth::get_particles()
{
    cudaError_t rc = cudaMemcpy(particles, dev_particles, sizeof(particle) * num_particles, 
                                cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess)
    {
        std::cout << "GPU Transfer to host failed" << std::endl;
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

            particles[i*num_particles_width + j].pos = {x, 1.0f, z};
            particles[i*num_particles_width + j].prev_pos = {x, 1.0f, z};
            particles[i*num_particles_width + j].force = {0.0f, 0.0f, 0.0f};
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int dev_num_particles_width = cuConstRendererParams.num_particles_width;
    int dev_num_particles_height = cuConstRendererParams.num_particles_height;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x = blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    particle *dev_particles = cuConstRendererParams.dev_particles;
    vector3D tot_force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    //cooperatively load into the array the necessary particles starting 
    //with those contained in the thread block.
    int idx = row * dev_num_particles_width + col;
    blk_particles[row + 1][col + 1] = dev_particles[idx];
    //for border particles, load the immediate top, left, bottom, and right
    //load tops so only if you're in the top row of your block
    if(threadIdx.y == 0)
    {
        //ensure you're not the top row in the system
        if(row != 0)
        {
            int top = (row - 1) * dev_num_particles_width + col;
            blk_particles[0][threadIdx.x + 1] = dev_particles[top];
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
        if(row != dev_num_particles_height - 1)
        {
            int btm = (row + 1) * dev_num_particles_width + col;
            blk_particles[threadIdx.y + 2][threadIdx.x + 1] = dev_particles[btm];
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
            blk_particles[threadIdx.y + 1][0] = dev_particles[left];
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
        if(col != dev_num_particles_width - 1)
        {
            int right = idx + 1;
            blk_particles[threadIdx.y + 1][threadIdx.x + 2] = dev_particles[right];
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
            int top_left = (row - 1) * dev_num_particles_width + (col - 1);
            blk_particles[0][0] = dev_particles[top_left];
        }
        else
        {
            blk_particles[0][0].pos = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //top right
    else if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
    {
        if(row != 0 && col != dev_num_particles_width - 1)
        {
            int top_right = (row - 1) * dev_num_particles_width + (col + 1);
            blk_particles[0][shared_mem_x - 1] = dev_particles[top_right];
        }
        else
        {
            blk_particles[0][shared_mem_x - 1].pos = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //bottom left
    else if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
    {
        if(row != dev_num_particles_height - 1 && col != 0)
        {
            int btm_left = (row + 1) * dev_num_particles_width + col - 1;
            blk_particles[shared_mem_y - 1][0] = dev_particles[btm_left];
        }
        else
        {
            blk_particles[shared_mem_y - 1][0].pos = vector3D(-2.0, -2.0, -2.0);
        }
    }
    //bottom right 
    else if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
    {
        if(row != dev_num_particles_height - 1 && col != dev_num_particles_width - 1)
        {
            int btm_right = (row + 1) * dev_num_particles_width + col + 1;
            blk_particles[shared_mem_y - 1][shared_mem_x - 1] = dev_particles[btm_right];
        }
    }
}

__global__ void apply_all_forces()
{
    //row in col within the entire grid of threads
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int dev_num_particles_width = cuConstRendererParams.num_particles_width;
    int dev_num_particles_height = cuConstRendererParams.num_particles_height;
    //create a 2D array of shared memory for particles that will be used by 
    //block
    int shared_mem_x =  blockDim.x + 2;
    int shared_mem_y = blockDim.y + 2;
    particle *dev_particles = cuConstRendererParams.dev_particles;
    vector3D force = vector3D(0.0f, -9.81f * PARTICLE_MASS, 0.0f);
    __shared__ particle blk_particles[TPB_Y + 2][TPB_X + 2];
    if(row < dev_num_particles_height && col < dev_num_particles_width)
    {
        load_particles((particle **)blk_particles);
    }
    __syncthreads();
}

void CudaCloth::apply_forces()
{
  //setup invocaiton for application of all forces
}

__global__ void update_all_positions()
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int dev_num_particles_width = cuConstRendererParams.num_particles_width;
    int dev_num_particles_height = cuConstRendererParams.num_particles_height;
    if(row < dev_num_particles_height && col < dev_num_particles_width)
    {
        int i = row * dev_num_particles_width + col;
        particle curr = cuConstRendererParams.dev_particles[i];
        vector3D temp(curr.pos);
        vector3D acc = curr.force/PARTICLE_MASS;
        curr.pos += (curr.pos - curr.prev_pos +
                             acc * TIME_STEP * TIME_STEP); 
        curr.prev_pos = temp;
        cuConstRendererParams.dev_particles[i] = curr;
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
    //satisfy_constraints();
}

