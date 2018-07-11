set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_BUILD_TYPE Debug)
IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(rnn_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 ")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)


option(WITH_OMP "compile iRNN with openmp." ON)

if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()

#prepare MKL lib
include(cmake/MKL.cmake)
include_directories(${MKLINC}) 
link_directories(${MKLLIBPATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${MKLLIBPATH}")

SET(WITH_CPU TRUE)
IF (WITH_CPU)
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(iRNN SHARED src/gru_xw_inference.cpp 
        src/gru_xw_training.cpp
        src/lstm_xw_inference.cpp
        src/lstm_xw_training.cpp)
    TARGET_LINK_LIBRARIES(iRNN mkl_rt)


    add_executable(rnn_bench tests/benchmark.cpp)
    TARGET_LINK_LIBRARIES(rnn_bench iRNN)
    SET_TARGET_PROPERTIES(rnn_bench PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS iRNN
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/rnn.h DESTINATION "include")

ENDIF()


