#include <iostream>
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        cout << "Hello World!" << endl;
    }
    Kokkos::finalize();
    return 0;
}
