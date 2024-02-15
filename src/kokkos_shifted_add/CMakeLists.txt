set(exec_name kokkos_shifted_add_local_view)
add_example(${exec_name}.cpp)
find_package(KokkosRemoteSpaces REQUIRED)
target_link_libraries(${exec_name} Kokkos::kokkos
  Kokkos::kokkosremotespaces)
target_include_directories(${exec_name} PRIVATE ${KokkosKernels_SOURCE_DIR})
target_compile_features(${exec_name} PRIVATE cxx_std_23)
