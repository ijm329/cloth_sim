#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glx.h>
#endif

#include "cycleTimer.h"
#include "cloth.h"

#define DEFAULT_W 640
#define DEFAULT_H 480

Cloth cloth(64);

//mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

void render_scene()
{
    double start_time, end_time;
    
    //start_time = CycleTimer::currentSeconds();
    cloth.render(rotate_x, rotate_y, translate_z);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
    //glTranslatef(0.0f, 0.0f, translate_z);
    //glRotatef(rotate_x, 1.0f, 0.0f, 0.0f);
    //glRotatef(rotate_y, 0.0f, 1.0f, 0.0f);

    //glColor3f(0.0f, 1.0f, 0.0f);
    //glBegin(GL_POINTS);
    //glVertex3f(0.0f, 0.0f, 0.0f);
    //glVertex3f(1.0f, 0.0f, 0.0f);
    //glVertex3f(0.0f, 0.0f, 1.0f);
    //glVertex3f(1.0f, 0.0f, 1.0f);
    //glEnd();

    glutSwapBuffers();

    glutPostRedisplay();
    //end_time = CycleTimer::currentSeconds();

    //std::cout << "Render Time: " << end_time - start_time << std::endl;
}

void resize_window(int width, int height)
{
    printf("Resizing window \n");
    height = (height == 0) ? 1 : height;
    float aspect_ratio = (GLfloat) width / (GLfloat) height;
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0f, aspect_ratio, 0.1f, 10.0f);
}

void mouse_handler(int button, int state, int x, int y)
{
    printf("button = %d \n", button);
    if((button == 3) || (button == 4))
    {
        if(state == GLUT_UP)
            return;
        if(button == 3)
            translate_z += 0.02f;
        else
            translate_z -= 0.02f;
    }
    else
    {
        if(state == GLUT_DOWN)
            mouse_buttons |= 1<<button;
        else if(state == GLUT_UP)
            mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void move_camera(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if(mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void process_keys(unsigned char key, int x, int y)
{
    //x and y are mouse position relative to top left corner when key is pressed
    if((key == 'q') || (key == 'Q') || (key == 27))
        exit(0);
}

void printVersionInfo()
{
    std::cout << " Glew version: " << glewGetString(GLEW_VERSION) <<
               " OpenGL version: " <<glGetString(GL_VERSION) << std::endl;
}

void glInit(int argc, char **argv)
{
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

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glClearDepth(1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(1.0);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, DEFAULT_W, DEFAULT_H);
}

int main(int argc, char **argv)
{
    //initialize GLUT and create window
    glInit(argc, argv);
    cloth.init();

    //register GLUT callbacks
    glutDisplayFunc(render_scene);
    glutReshapeFunc(resize_window);
    glutKeyboardFunc(process_keys);
    glutMouseFunc(mouse_handler);
    glutMotionFunc(move_camera);

    //enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
