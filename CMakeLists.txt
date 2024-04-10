cmake_minimum_required(VERSION 3.5)

project(Parallel_Computing_Basics-Practice_9-8_semester LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Parallel_Computing_Basics-Practice_9-8_semester main.cpp)

include(GNUInstallDirs)
install(TARGETS Parallel_Computing_Basics-Practice_9-8_semester
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(Parallel_Computing_Basics-Practice_9-8_semester PRIVATE ${OpenMP_CXX_FLAGS})
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Parallel_Computing_Basics-Practice_9-8_semester ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})
