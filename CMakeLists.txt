cmake_minimum_required(VERSION 3.16)
project(kokkos_playground)

# found from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
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
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/kokkos/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/taco/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(kokkos)
add_subdirectory(taco)

add_executable(kokkos_playground main.cpp)
target_include_directories(kokkos_playground PRIVATE
  ${CMAKE_SOURCE_DIR}/taco/include
  ${CMAKE_BINARY_DIR}/taco/include)
target_link_libraries(kokkos_playground Kokkos::kokkos taco)
