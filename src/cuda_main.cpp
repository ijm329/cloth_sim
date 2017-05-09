#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <assert.h>

#include "cycleTimer.h"
#include "cuda_cloth.h"

/*#define DEFAULT_W 640
#define DEFAULT_H 480
#define REFRESH_INTERVAL 10 //in ms*/
#define NUM_CLOTH_POINTS 40

CudaCloth cuda_cloth(NUM_CLOTH_POINTS);


int main(int argc, char **argv)
{
    cuda_cloth.init();
    cuda_cloth.simulate_timestep();
    cuda_cloth.get_particles();
    return 0;
}
