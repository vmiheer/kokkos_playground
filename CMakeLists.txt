cmake_minimum_required(VERSION 3.16)
project(kokkos_playground)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# found from
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with
        ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

function (add_example NAME)
  cmake_path(GET NAME STEM EXEC_NAME)
  message(STATUS "Adding example ${EXEC_NAME}")
  add_executable(${EXEC_NAME} ${NAME})
  target_link_libraries(${EXEC_NAME} Kokkos::kokkos fmt::fmt)
  target_compile_features(${EXEC_NAME} PRIVATE cxx_std_23)
endfunction()

# Libs we link against
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
set(KokkosKernels_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kokkos-kernels)
add_subdirectory(taco)
add_subdirectory(fmt)

add_subdirectory(src)

add_example(taco_example.cpp)
target_include_directories(taco_example PRIVATE
  ${CMAKE_SOURCE_DIR}/taco/include
  ${CMAKE_BINARY_DIR}/taco/include)
target_link_libraries(taco_example taco)

add_example(subview.cpp)
