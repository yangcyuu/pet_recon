cmake_minimum_required(VERSION 3.10)
project(pni)
set(${PROJECT_VERSION} "1.0")
execute_process(
  COMMAND pwd
  OUTPUT_VARIABLE SYSTEM_CURRENT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(PROJECT_CONFIG_HPP ${PROJECT_SOURCE_DIR}/include/PnI-Config.hpp)

file(GLOB_RECURSE SOURCE_CODE_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SOURCE_CODE_CUDA ${PROJECT_SOURCE_DIR}/src/*.cu)
set(SOURCE_CODE_ALL ${SOURCE_CODE_CPP} ${SOURCE_CODE_CUDA})
file(GLOB_RECURSE INCLUDE_LIST ${PROJECT_SOURCE_DIR}/include/*.hpp
     ${PROJECT_SOURCE_DIR}/include/*.cuh)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${CUDA_CCCL_INCLUDE_DIR})
include_directories(${CUDA_LINUX_INCLUDE_DIR})

set(APP_DIR ${PROJECT_SOURCE_DIR}/app)
set(RELEASE_DIR ${PROJECT_SOURCE_DIR}/release)

function(found_macro_in_file IN_FILE IN_STRING OUT_VAR)
  file(READ ${IN_FILE} FILE_CONTENTS)
  string(FIND "${FILE_CONTENTS}" "${IN_STRING} 1" FIND_INDEX)
  if(FIND_INDEX EQUAL -1)
    set(${OUT_VAR}
        0
        PARENT_SCOPE)
  else()
    set(${OUT_VAR}
        1
        PARENT_SCOPE)
  endif()
endfunction()

function(append VAR_NAME STRING_TO_APPEND)
  set(CURRENT_VALUE "${${VAR_NAME}}")
  set(${VAR_NAME}
      "${CURRENT_VALUE} ${STRING_TO_APPEND}"
      PARENT_SCOPE)
endfunction()
function(append_line VAR_NAME STRING_TO_APPEND)
  set(CURRENT_VALUE "${${VAR_NAME}}")
  set(${VAR_NAME}
      "${CURRENT_VALUE}\n${STRING_TO_APPEND}"
      PARENT_SCOPE)
endfunction()

found_macro_in_file(${PROJECT_CONFIG_HPP} "PNI_STANDARD_CONFIG_DISABLE_CUDA"
                     PNI_STANDARD_CONFIG_DISABLE_CUDA)
found_macro_in_file(${PROJECT_CONFIG_HPP} "PNI_STANDARD_CONFIG_ENABLE_DPDK"
                     PNI_STANDARD_CONFIG_ENABLE_DPDK)
if(PNI_STANDARD_CONFIG_DISABLE_CUDA EQUAL 0)
  message(STATUS "CUDA is enabled")
  set(CUDA_PNI_LIB "${PROJECT_NAME}_cu")
  set(CUDA_PNI_LINK_FLAGS
      "-l${CUDA_PNI_LIB} -L${CUDA_LIBRARY_DIR} -lcudart -lcufft")
  set(CUDA_LIB "-L${CUDA_LIBRARY_DIR} -lcudart")
  set(CUDA_INCLUDE "-I${CUDA_INCLUDE_DIR} -I${CUDA_LINUX_INCLUDE_DIR}")
endif()

set(APP_AQST "pni-aqst")
file(GLOB_RECURSE SRC_AQST ${APP_DIR}/acquisition/acquisition.cpp)
set(APP_RAWDATA "pni-rawdata")
file(GLOB_RECURSE SRC_RAWDATA ${APP_DIR}/acquisition/rawfile.cpp)

function(make_pkgconfig)
  set(PKGCONFIG_CONTENT "prefix=${CMAKE_INSTALL_PREFIX}")
  append_line(PKGCONFIG_CONTENT
              "includedir=\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
  append_line(PKGCONFIG_CONTENT "libdir=\${prefix}/${CMAKE_INSTALL_LIBDIR}")
  append_line(PKGCONFIG_CONTENT "")
  append_line(PKGCONFIG_CONTENT "Name: lib${PROJECT_NAME}")
  append_line(PKGCONFIG_CONTENT
              "Description: The plugin & image standard library.")
  append_line(PKGCONFIG_CONTENT "Version: ${PROJECT_VERSION}")
  append_line(PKGCONFIG_CONTENT
              "Libs: -L\${libdir} -lpni ${CUDA_PNI_LINK_FLAGS}  ")
  append_line(
    PKGCONFIG_CONTENT
    "Cflags: -I\${includedir} -include \${includedir}/${PROJECT_NAME}/PnI-Config.hpp ${CUDA_INCLUDE}"
  )
  if(PNI_STANDARD_CONFIG_ENABLE_DPDK EQUAL 1)
    append_line(PKGCONFIG_CONTENT "Requires: libdpdk")
  endif()
  append_line(PKGCONFIG_CONTENT "Requires: libevent")
  execute_process(COMMAND echo ${PKGCONFIG_CONTENT} OUTPUT_FILE ./build/libpni.pc)
  install(FILES ./build/libpni.pc
          DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endfunction()

set(OPTIMIZE "-O2")
set(COMPILE_DEBUG "-g")
set(GLOBAL_INCLUDE "-include ${PROJECT_SOURCE_DIR}/include/PnI-Config.hpp")

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  # set(CMAKE_CXX_COMPILER "g++-13")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${GLOBAL_INCLUDE} ${OPTIMIZE} -Wreturn-type -Wnoexcept ${COMPILE_DEBUG}"
  )
  set(CMAKE_C_FLAGS
      "${CMAKE_C_FLAGS} ${GLOBAL_INCLUDE} ${OPTIMIZE} -Wreturn-type ${COMPILE_DEBUG}"
  )
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} ${GLOBAL_INCLUDE} ${OPTIMIZE} -Wreturn-type ${COMPILE_DEBUG} -Wnoexcept -std=c++20 --extended-lambda"
  )
  set(CMAKE_CXX_STANDARD 23)
  # set(CMAKE_CXX_STANDARD_REQUIRED ON)

  include(GNUInstallDirs)
  if(PNI_STANDARD_CONFIG_ENABLE_DPDK EQUAL 1)
    execute_process(
      COMMAND pkg-config --cflags libdpdk
      OUTPUT_VARIABLE INCLUDE_DPDK
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND pkg-config --libs libdpdk
      OUTPUT_VARIABLE LINK_DPDK
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INCLUDE_DPDK}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${INCLUDE_DPDK}")
  set(DPDK_LINK_FLAGS ${LINK_DPDK})
  add_library(${PROJECT_NAME} SHARED ${SOURCE_CODE_CPP})

  if(PNI_STANDARD_CONFIG_DISABLE_CUDA EQUAL 0)
    # Automatically detect and set CUDA architectures
    include(CheckLanguage)
   
    if(CMAKE_CUDA_COMPILER)
     set(CMAKE_CUDA_ARCHITECTURES 80 86)
      # enable_language(CUDA)
      # Set CUDA architectures; update this list as needed for your GPUs
     
    endif()
     check_language(CUDA)
    enable_language(CUDA)
    add_library(${CUDA_PNI_LIB} SHARED ${SOURCE_CODE_CUDA})
    install(TARGETS ${CUDA_PNI_LIB} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY "include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pni)

  set(LINK_LIBS ${PROJECT_NAME} ${DPDK_LINK_FLAGS} ${CUDA_PNI_LIB} ${CUDA_LIB})

  # 找到 libevent
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBEVENT REQUIRED libevent)

  # 包含头文件
  include_directories(${LIBEVENT_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${LIBEVENT_LIBRARIES})
  target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
  add_executable(${APP_AQST} ${SRC_AQST})
  target_link_libraries(${APP_AQST} ${LINK_LIBS} ${LIBEVENT_LIBRARIES})
  install(TARGETS ${APP_AQST} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(${APP_RAWDATA} ${SRC_RAWDATA})
  target_link_libraries(${APP_RAWDATA} ${LINK_LIBS} ${LIBEVENT_LIBRARIES})
  install(TARGETS ${APP_RAWDATA} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  make_pkgconfig()

  # # 查找 FFTW 库 find_package(FFTW REQUIRED)

  # if (FFTW_FOUND) include_directories(${FFTW_INCLUDE_DIRS}) list(APPEND
  # LINK_LIBS ${FFTW_LIBRARIES}) else() message(FATAL_ERROR "FFTW library not
  # found!") endif()

  # # 在目标中链接 FFTW target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
  set(CMAKE_CXX_COMPILER
      "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.43.34808\\bin\\Hostx64\\x64\\cl.exe"
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /Ox /W3 /EHsc")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox /W3 /EHsc")

  include_directories("D:\\PnI\\boost_1_87_0")
  include_directories("D:\\PnI\\cxxopts\\include")
  include_directories("D:\\PnI\\termcolor\\include")

  set(EXECUTABLE_OUTPUT_PATH ${RELEASE_DIR})

  add_library(${PROJECT_NAME} STATIC ${SOURCE_CODE_CPP})

  execute_process(COMMAND mkdir ${RELEASE_DIR})

  # install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  # install(FILES ${INCLUDE_LIST} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pni)

  add_executable(${APP_AQST} ${SRC_AQST})
  target_link_libraries(${APP_AQST} ${PROJECT_NAME} ${DPDK_LINK_FLAGS})
  install(TARGETS ${APP_AQST} RUNTIME DESTINATION ${RELEASE_DIR})

  add_executable(${APP_RAWDATA} ${SRC_RAWDATA})
  target_link_libraries(${APP_RAWDATA} ${PROJECT_NAME} ${DPDK_LINK_FLAGS})
  install(TARGETS ${APP_RAWDATA} RUNTIME DESTINATION ${RELEASE_DIR})

endif()
