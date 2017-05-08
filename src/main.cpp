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
#include "cloth.h"

#define DEFAULT_W 640
#define DEFAULT_H 480
#define REFRESH_INTERVAL 10 //in ms
#define NUM_CLOTH_POINTS 3

Cloth cloth(NUM_CLOTH_POINTS);

//mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = -45.0;
float translate_z = MIN_BOUND*3;

void render_scene()
{
    double start_time, end_time;
    
    start_time = CycleTimer::currentSeconds();

    //cloth.simulate_timestep();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    cloth.render(rotate_x, rotate_y, translate_z);

    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
    //glTranslatef(0.0f, 0.0f, translate_z);
    //glRotatef(rotate_x, 1.0f, 0.0f, 0.0f);
    //glRotatef(rotate_y, 0.0f, 1.0f, 0.0f);

    //glColor3f(0.0f, 1.0f, 0.0f);
    //glBegin(GL_POINTS);
    //glVertex3f(-1.0f, 0.0f, -1.0f);
    //glVertex3f(0.0f, 0.0f, -1.0f);
    //glVertex3f(0.0f, 0.0f, 0.0f);
    //glVertex3f(-1.0f, 0.0f, 0.0f);

    //glEnd();

    glutSwapBuffers();
    end_time = CycleTimer::currentSeconds();
    char window_title[256];
    sprintf(window_title, "cloth_sim FPS: %f", 1/(end_time - start_time));
    glutSetWindowTitle(window_title);
}

void timer_handler(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_INTERVAL, timer_handler, 0);
    }
}

void resize_window(int width, int height)
{
    height = (height == 0) ? 1 : height;
    float aspect_ratio = (GLfloat) width / (GLfloat) height;
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0f, aspect_ratio, 0.1f, 100.0f);
}

void mouse_handler(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
        if(button == 3)
            translate_z += 0.1f;
        else if(button == 4)
            translate_z -= 0.1f;
    }

    else if(state == GLUT_UP)
        mouse_buttons = 0;

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
    else if(mouse_buttons & 4)
        translate_z += dy * 0.01f;

    mouse_old_x = x;
    mouse_old_y = y;
}

void process_keys(unsigned char key, int x, int y)
{
    //x and y are mouse position relative to top left corner when key is pressed
    if((key == 'q') || (key == 'Q') || (key == 27))
        exit(0);
    else if(key == 32)
        cloth.simulate_timestep();
    else if((key == 'r') || (key == 'R'))
        cloth.init();
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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutCreateWindow("cloth_sim");

    //glew init
    if(glewInit() != GLEW_OK)
    {
        std::cout<<"Error: could not initialize GLEW!"<<std::endl;
        exit(1);
    }
    printVersionInfo();

    glClearColor(0.0, 0.0, 0.0, 1.0);

    //smoothing
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    //blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //lighting
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_pos[4] = {0.0, 2.0, 0.0, 0.0};
    GLfloat light_diffuse[3] = {1.0, 1.0, 1.0};
    GLfloat light_ambient[3] = {0.0f, 0.0f, 0.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    //material properties
    GLfloat cloth_shininess[1] = {100.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, cloth_shininess);
    
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_MULTISAMPLE);
    glPointSize(2.0);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glViewport(0, 0, DEFAULT_W, DEFAULT_H);
}

void vector3Dtest()
{
    vector3D u(1,2,3);
    vector3D v(u);
    assert(u == v);
    std::cout<<(u+v)<<std::endl;
    std::cout<<(u-v)<<std::endl;
    std::cout<<(u*5)<<5*u<<std::endl;
    assert(u*5 == 5*u);
    u*=5;
    u+=v;
    std::cout<<(u)<<std::endl;
    u-=v;
    std::cout<<(u)<<std::endl;
    std::cout<<(u/5)<<std::endl;
    assert(u/5 == v);
}

int main(int argc, char **argv)
{
    //initialize GLUT and create window
    glInit(argc, argv);
    cloth.init();

    //vector3Dtest();

    //register GLUT callbacks
    glutDisplayFunc(render_scene);
    glutReshapeFunc(resize_window);
    glutTimerFunc(REFRESH_INTERVAL, timer_handler, 0);
    glutKeyboardFunc(process_keys);
    glutMouseFunc(mouse_handler);
    glutMotionFunc(move_camera);

    ////enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
