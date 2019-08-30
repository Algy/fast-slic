cmake_minimum_required (VERSION 3.0)
project (fast-slic LANGUAGES CXX)


set (CMAKE_CXX_STANDARD 11)
#set (CMAKE_BUILD_TYPE RelWithDebugInfo)
set (CMAKE_BUILD_TYPE Debug)
find_package(GTest REQUIRED)
enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(run_tests src/cca.cpp src/cpptest/main.cpp src/cpptest/test_cca.cpp src/cpptest/test_cielab.cpp)
target_link_libraries(run_tests ${GTEST_LIBRARIES} pthread)
