#include <iostream>
#include <Kokkos_Core.hpp>
#include <taco.h>

using namespace std;

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        auto a = taco::read("494_bus.mtx", taco::COO(2));
        cout << "Hello World!" << endl;
    }
    Kokkos::finalize();
    return 0;
}
