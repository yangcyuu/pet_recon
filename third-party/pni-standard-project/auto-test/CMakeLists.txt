# 查找 Google Test
project(test)
cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)    
find_package(Threads REQUIRED) 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
execute_process(COMMAND pkg-config --libs --cflags libpni
    OUTPUT_VARIABLE PKG_CONFIG_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PKG_CONFIG_FLAGS}")


message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# 添加测试源文件
message(STATUS "Searching for test source files in ${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

execute_process(
    COMMAND pwd
    OUTPUT_VARIABLE CURRENT_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories("${CURRENT_PATH}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# 使用 pkg-config 查找 libpni

# 创建测试目标
add_executable(runTests ${TEST_SRC})



# 链接 Google Test 和 pthread 库
target_link_libraries(runTests GTest::GTest GTest::Main pthread)

# 添加测试
add_test(NAME runTests COMMAND runTests --gtest-break-on-failure)