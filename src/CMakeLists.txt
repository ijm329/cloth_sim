cmake_minimum_required (VERSION 2.8)

file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CUDA_SOURCES "*.cu")

include_directories(${CUDA_INCLUDE_DIRS}
                    ${GLEW_INCLUDE_DIRS}
                    ${GLUT_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

if(${CUDA_FOUND})
    cuda_include_directories(${GLEW_INCLUDE_DIRS}
                             ${GLUT_INCLUDE_DIRS}
                             ${OPENGL_INCLUDE_DIRS})
    cuda_include_directories(${PROJECT_SOURCE_DIR}/include)
    cuda_add_executable(cloth_sim_cuda ${CUDA_SOURCES})
    target_link_libraries(cloth_sim_cuda
                          ${GLEW_LIBRARIES}
                          ${GLUT_LIBRARIES}
                          ${OPENGL_LIBRARIES})
endif(${CUDA_FOUND})

add_executable(cloth_sim ${CPP_SOURCES})
target_link_libraries(cloth_sim
                      ${GLEW_LIBRARIES} 
                      ${GLUT_LIBRARIES} 
                      ${OPENGL_LIBRARIES} 
                      ${FREETYPE_LIBRARIES} 
                      ${CMAKE_THREADS_INIT})

set(EXECUTABLE_OUTPUT_PATH ..)
