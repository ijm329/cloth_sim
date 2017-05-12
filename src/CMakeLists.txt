cmake_minimum_required (VERSION 2.8)

file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CUDA_SOURCES "*.cu")

include_directories(${GLEW_INCLUDE_DIRS}
                    ${GLUT_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

if(${CUDA_FOUND})
    include_directories(${CUDA_INCLUDE_DIRS})
    cuda_include_directories(${GLEW_INCLUDE_DIRS}
                             ${GLUT_INCLUDE_DIRS}
                             ${OPENGL_INCLUDE_DIRS}
                             ${PROJECT_SOURCE_DIR}/include)
    #cuda_add_executable(cuda_cloth ${CUDA_SOURCE} cuda_cloth.cu cuda_main.cpp)
    cuda_add_executable(cloth_sim ${CUDA_SOURCE} cuda_cloth.cu vector3D.cpp cloth.cpp main.cpp)
    target_link_libraries(cloth_sim
                          ${GLEW_LIBRARIES}
                          ${GLUT_LIBRARIES}
                          ${OPENGL_LIBRARIES})
endif(${CUDA_FOUND})

#add_executable(cloth_sim ${CPP_SOURCE} vector3D.cpp cloth.cpp main.cpp)
#target_link_libraries(cloth_sim
#                      ${GLEW_LIBRARIES} 
#                      ${GLUT_LIBRARIES} 
#                      ${OPENGL_LIBRARIES} 
#                      ${CMAKE_THREADS_INIT})

set(EXECUTABLE_OUTPUT_PATH ..)
