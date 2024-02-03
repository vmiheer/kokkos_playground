#include "Kokkos_Core_fwd.hpp"
#include "fmt/core.h"
#include <Kokkos_Core.hpp>
#include <fmt/format.h>
#include <type_traits>

template <class DataTy, class... Args>
struct fmt::formatter<Kokkos::View<DataTy *, Args...>> {
  using U = Kokkos::View<DataTy *, Args...>;
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const U &a, FormatContext &ctx) {
    fmt::format_to(ctx.out(), "{{");
    for (auto i = 0; i < a.extent(0); i++) {
      fmt::format_to(ctx.out(), "{}", a(i));
      if (i < a.extent(0) - 1) {
        fmt::format_to(ctx.out(), ", ");
      }
    }
    return fmt::format_to(ctx.out(), "}}");
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using namespace Kokkos;
    auto a = View<double *>("a", 10);
    parallel_for(
        "init", a.extent(0), KOKKOS_LAMBDA(auto i) { a(i) = float(i + 1); });
    fmt::println("{}", a);

    auto b = subview(a, std::pair(0, 4));
    fmt::println("{}", b);
  }
  Kokkos::finalize();
  return 0;
}
