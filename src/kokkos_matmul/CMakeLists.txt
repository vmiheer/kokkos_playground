add_example(kokkos_matmul.cpp)
find_package(KokkosRemoteSpaces REQUIRED)
target_link_libraries(kokkos_matmul Kokkos::kokkos Kokkos::kokkosremotespaces)
target_include_directories(kokkos_matmul PRIVATE ${KokkosKernels_SOURCE_DIR})
