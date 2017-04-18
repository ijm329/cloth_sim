cmake_minimum_required (VERSION 2.8)

set(SOURCES main.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(cloth_sim ${SOURCES})
target_link_libraries(cloth_sim
                      ${GLEW_LIBRARIES} 
                      ${GLUT_LIBRARIES} 
                      ${OPENGL_LIBRARIES} 
                      ${FREETYPE_LIBRARIES} 
                      ${CMAKE_THREADS_INIT})

set(EXECUTABLE_OUTPUT_PATH ..)
