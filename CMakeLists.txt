cmake_minimum_required(VERSION 3.23)
project(kokkos_playground)
add_subdirectory(kokkos)

add_executable(kokkos_playground main.cpp)
target_link_libraries(kokkos_playground Kokkos::kokkos)
