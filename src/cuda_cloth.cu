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

__global__ void apply_all_forces()
{
}

void CudaCloth::apply_forces()
{

}

void CudaCloth::simulate_timestep()
{
    apply_forces();
    update_positions();
    satisfy_constraints();
}

