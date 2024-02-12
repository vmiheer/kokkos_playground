// clang-format off
#include <fstream>

#include "Kokkos_Core.hpp"

#include "fmt/core.h"
#include <fmt/format.h>
#include <Kokkos_RemoteSpaces.hpp>

// clang-format on

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
  constexpr size_t M = 1024;
  Kokkos::initialize(argc, argv);
  {
    using namespace Kokkos;
    auto a = View<double **>("a", M, M);
    auto b = View<double **>("b", M, M);
    auto c = View<double **>("c", M, M);
    parallel_for(
        "init", MDRangePolicy<Rank<2>>({0, 0}, {a.extent(0), a.extent(1)}),
        KOKKOS_LAMBDA(auto i, auto j) {
          a(i, j) = float(j + i * a.extent(0));
          b(i, j) = (i == j) ? 1. : 0.;
          c(i, j) = 0.;
        });
    if (M < 8) {
      fmt::print("{}", a);
      fmt::print("{}", b);
    }
    parallel_for(
        "matmul", MDRangePolicy<Rank<2>>({0, 0}, {c.extent(0), c.extent(1)}),
        KOKKOS_LAMBDA(auto i, auto j) {
          parallel_reduce(
              "matmul", a.extent(1),
              KOKKOS_LAMBDA(auto k, double &tmp) { tmp += a(i, k) * b(k, j); },
              c(i, j));
        });
    fmt::print("{}\n", c(64, 64));
  }
  Kokkos::finalize();

  return 0;
}
