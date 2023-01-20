cmake_minimum_required(VERSION 3.16)
project(kokkos_playground)
add_subdirectory(kokkos)
add_subdirectory(taco)

add_executable(kokkos_playground main.cpp)
target_include_directories(kokkos_playground PRIVATE
  ${CMAKE_SOURCE_DIR}/taco/include
  ${CMAKE_BINARY_DIR}/taco/include)
target_link_libraries(kokkos_playground Kokkos::kokkos taco)
