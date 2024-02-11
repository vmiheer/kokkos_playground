#!/usr/bin/env zsh
export WORKSPACE=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
export Kokkos_ROOT=$WORKSPACE/kokkos_install
export CMAKE_PREFIX_PATH=$Kokkos_ROOT:$CMAKE_PREFIX_PATH

git submodule update --init --recursive

cd $WORKSPACE/kokkos
mkdir -p build && cd build
if [[ ! -f build/CMakeCache.txt ]]; then
  cmake -DCMAKE_INSTALL_PREFIX=$Kokkos_ROOT -DBUILD_SHARED_LIBS=ON ..
fi
cmake --build . -j 24 --target install

cd $WORKSPACE/kokkos-kernels
mkdir -p build && cd build
if [[ ! -f build/CMakeCache.txt ]]; then
  cmake -DCMAKE_INSTALL_PREFIX=$Kokkos_ROOT ..
fi
cmake --build . -j 24 --target install

cd $WORKSPACE/kokkos-remote-spaces
mkdir -p build && cd build
if [[ ! -f build/CMakeCache.txt ]]; then
  cmake -DCMAKE_INSTALL_PREFIX=$Kokkos_ROOT -DKRS_ENABLE_MPISPACE=ON ..
fi
cmake --build . -j 24 --target install
cd $WORKSPACE
