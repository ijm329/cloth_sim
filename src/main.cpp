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
    static float angle = 0.0f;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 10.0f,
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
    glRotatef(angle, 0.0f, 1.0f, 0.0f);

    glColor3f(0.2f, 0.2f, 0.4f);

    glBegin(GL_TRIANGLES);
    glVertex3f(-2.0, -2.0, -5.0);
    glVertex3f(2.0, 0.0, -5.0);
    glVertex3f(0.0, 2.0, -5.0);
    glEnd();

    angle += 0.1f;

    glutSwapBuffers();
}

void resizeWindow(int width, int height)
{
    height = (height == 0) ? 1 : height;
    float ratio = 1.0 * width / height;
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glViewport(0, 0, width, height);

    gluPerspective(45, ratio, 1, 1000);

    glMatrixMode(GL_MODELVIEW);
}

void processKeys(unsigned char key, int x, int y)
{
    //x and y are mouse position relative to top left corner when key is pressed
    if((key == 'q') || (key == 'Q') || (key == 27))
        exit(0);
}

void printVersionInfo()
{
    std::cout << " Glew version: " << glewGetString(GLEW_VERSION) <<
               " GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) <<
               " OpenGL version: " <<glGetString(GL_VERSION) << std::endl;
}

void glInit()
{
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glClearColor(0.0, 0.0, 0.0, 0.0);
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

    glInit();

    //register GLUT callbacks
    glutDisplayFunc(renderScene);
    glutReshapeFunc(resizeWindow);
    glutIdleFunc(renderScene);
    glutKeyboardFunc(processKeys);

    //enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
