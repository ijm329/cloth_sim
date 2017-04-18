#include <iostream>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <math.h>

#define DEFAULT_W 960
#define DEFAULT_H 640

void renderScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(0.1, 0.2, 0.3);

    glBegin(GL_TRIANGLES);
    glVertex3f(0.0, 0.5, 0.0);
    glVertex3f(-0.5, -0.5, 0.0);
    glVertex3f(0.5, -0.5, 0.0);
    glEnd();

    glutSwapBuffers();
}

int main(int argc, char **argv)
{

    //initialize GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(DEFAULT_W, DEFAULT_H);
    glutCreateWindow("cloth_sim");

    //glew init
    if(glewInit() != GLEW_OK)
    {
        std::cout<<"Error: could not initialize GLEW!"<<std::endl;
        exit(1);
    }
    std::cout<<" Glew version: " << glewGetString(GLEW_VERSION) <<
               " GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << 
               " OpenGL version: " <<glGetString(GL_VERSION) << std::endl;

    //register GLUT callbacks
    glutDisplayFunc(renderScene);

    //enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
