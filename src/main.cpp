#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define DEFAULT_W 960
#define DEFAULT_H 640

void renderScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(0.2f, 0.2f, 0.4f, 0.5f);

    glBegin(GL_TRIANGLES);
    glVertex3f(0.0, 0.5, 0.0);
    glVertex3f(-0.5, -0.5, 0.0);
    glVertex3f(0.5, -0.5, 0.0);
    glEnd();

    glutSwapBuffers();
}

void printVersionInfo()
{
    std::cout << " Glew version: " << glewGetString(GLEW_VERSION) <<
               " GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) <<
               " OpenGL version: " <<glGetString(GL_VERSION) << std::endl;
}

int main(int argc, char **argv)
{

    //initialize GLUT and create window
    glutInit(&argc, argv);
    glutInitWindowSize(DEFAULT_W, DEFAULT_H);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("cloth_sim");

    //glew init
    if(glewInit() != GLEW_OK)
    {
        std::cout<<"Error: could not initialize GLEW!"<<std::endl;
        exit(1);
    }
    printVersionInfo();

    //register GLUT callbacks
    glutDisplayFunc(renderScene);

    //enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
