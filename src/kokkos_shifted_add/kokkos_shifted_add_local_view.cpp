// clang-format off
#include <fstream>
#include <algorithm>
#include <numeric>
#include <ranges>

#include "fmt/core.h"
#include <fmt/format.h>

// #include "Kokkos_Core.hpp"
#include <Kokkos_RemoteSpaces.hpp>
// #include <Kokkos_RemoteSpaces_ViewLayout.hpp>
// #include <Kokkos_RemoteSpaces_Helpers.hpp>
// #include <impl/Kokkos_HostThreadTeam.hpp>

// clang-format on

using fmt::println;

template <class DataTy, class... Args>
struct fmt::formatter<Kokkos::View<DataTy, Args...>> {
  using U = Kokkos::View<DataTy, Args...>;
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const U &a, FormatContext &ctx) {
    if constexpr (U::rank == 2) {
      fmt::format_to(ctx.out(), "{{");
      for (auto i = 0; i < a.extent(0); ++i) {
        fmt::format_to(ctx.out(), "{{");
        for (auto j = 0; j < a.extent(1); ++j) {
          fmt::format_to(ctx.out(), "{}", a(i, j));
          if (j < a.extent(1) - 1) {
            fmt::format_to(ctx.out(), ", ");
          }
        }
        fmt::format_to(ctx.out(), "}}\n");
      }
      return fmt::format_to(ctx.out(), "");
    } else {
      static_assert(U::rank == 1);
      fmt::format_to(ctx.out(), "{{");
      for (auto i = 0; i < a.extent(0); i++) {
        fmt::format_to(ctx.out(), "{}", a(i));
        if (i < a.extent(0) - 1) {
          fmt::format_to(ctx.out(), ", ");
        }
      }
      return fmt::format_to(ctx.out(), "}}\n");
    }
  }
};

int main(int argc, char *argv[]) {
  using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
  constexpr size_t M = 8;
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
  if (!(mpi_thread_level_available >= mpi_thread_level_required)) {
    // if asserts are disabled, don't want to move forward.
    fmt::println(
        "mpi_thread_level_available >= mpi_thread_level_required failed");
    exit(1);
  }

  Kokkos::initialize(argc, argv);
  {
    using namespace Kokkos;
    using PartitionedView1D =
        Kokkos::View<double **, PartitionedLayoutRight, RemoteSpace_t>;
    using Local1DView = typename PartitionedView1D::HostMirror;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      println("MPI_COMM_WORLD size: {}", size);

    auto A = PartitionedView1D("RemoteView", size, M);
    auto Alocal = Local1DView("LocalView", 1, M);
    auto lr = Experimental::get_local_range(M);
    parallel_for(
        "init", (A.extent(1)),
        KOKKOS_LAMBDA(auto i) { A(rank, i) = rank * M + i; });

    for (auto i : std::ranges::iota_view(0, int(rank == 0 ? 1 : 0) * size)) {
      fmt::print("MPI_COMM_WORLD rank: {}: ", i);
      auto range = std::make_pair(size_t(0), M);
      for (auto j : std::ranges::iota_view(range.first, range.second))
        fmt::print("{}, ", double(A(i, j)));
      fmt::println("");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
