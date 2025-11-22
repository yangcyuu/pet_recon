# 查找 Google Test
project(test)
cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)    
find_package(Threads REQUIRED) 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

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
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPNI REQUIRED libpni)
include_directories(${LIBPNI_INCLUDE_DIRS})
link_directories(${LIBPNI_LIBRARY_DIRS})
add_definitions(${LIBPNI_CFLAGS_OTHER})

# 创建测试目标
add_executable(runTests ${TEST_SRC})



# 链接 Google Test 和 pthread 库
target_link_libraries(runTests  GTest::GTest GTest::Main pthread ${LIBPNI_LIBRARIES})

# 添加测试
add_test(NAME runTests COMMAND runTests --gtest-break-on-failure)