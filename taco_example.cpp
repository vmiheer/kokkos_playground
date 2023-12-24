#include <Kokkos_Core.hpp>
#include <iostream>
#include <taco.h>

using namespace std;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    assert(argc == 2);
    auto input_file = argv[1];
    auto a = taco::read(input_file, taco::COO(2));
    cout << a << endl;
  }
  Kokkos::finalize();
  return 0;
}
