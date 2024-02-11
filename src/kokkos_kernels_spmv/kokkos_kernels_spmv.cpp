// clang-format off
#include <fstream>

#include "Kokkos_Core.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "KokkosBatched_Util.hpp"

#include "fmt/core.h"
#include <fmt/format.h>

#include "perf_test/batched/sparse/KokkosBatched_Test_Sparse_Helper.hpp"
// clang-format on

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
    bool layout_left = true;
    bool layout_right = false;

    std::string name_A = "A.mm";
    std::string name_B = "B.mm";

    std::string name_timer = "timers";
    std::string name_X = "X";

    for (int i = 1; i < argc; ++i) {
      const std::string &token = argv[i];
      if (token == std::string("--help") || token == std::string("-h")) {
        std::cout
            << "Kokkos Batched SPMV performance test options:" << std::endl
            << "-A                :  Filename of the input batched matrix."
            << std::endl
            << "-B                :  Filename of the input batched right-hand "
               "side."
            << std::endl
            << "-X                :  Filename of the output batched solution."
            << std::endl
            << "-l                :  Specify left layout." << std::endl
            << "-r                :  Specify right layout." << std::endl
            << std::endl;
        return 0;
      }
      if (token == std::string("-A"))
        name_A = argv[++i];
      if (token == std::string("-B"))
        name_B = argv[++i];

      if (token == std::string("-X"))
        name_X = argv[++i];
    }

    int N, Blk, nnz, ncols;
    typedef Kokkos::LayoutRight LR;
    typedef Kokkos::LayoutLeft LL;

    using IntView = Kokkos::View<int *, LR>;
    using AMatrixValueViewLR = Kokkos::View<double *, LR>;
    using AMatrixValueViewLL = Kokkos::View<double *, LL>;
    using XYTypeLR = Kokkos::View<double **, LR>;
    using XYTypeLL = Kokkos::View<double **, LL>;

    readSizesFromMM(name_A, Blk, ncols, nnz, N);

    printf(" :::: Testing (N = %d, Blk = %d, nnz = %d)\n", N, Blk, nnz);
    IntView rowOffsets("values", Blk + 1);
    IntView colIndices("values", nnz);
    AMatrixValueViewLL valuesLL("values", N, nnz);
    XYTypeLL xLL("values", N, Blk);
    // readCRSFromMM(name_A, valuesLL, rowOffsets, colIndices);
    // fmt::print("{}", rowOffsets);
    // fmt::print("{}", colIndices);
    // fmt::print("{}", valuesLL);
    // readArrayFromMM(name_B, xLL);
    /*
    using alphaViewType = Kokkos::View<double *>;
    alphaViewType alphaV("alpha", N);
    alphaViewType betaV("alpha", N);

    IntView rowOffsets("values", Blk + 1);
    IntView colIndices("values", nnz);
    AMatrixValueViewLR valuesLR("values", N, nnz);
    AMatrixValueViewLL valuesLL("values", N, nnz);

    XYTypeLR xLR("values", N, Blk);
    XYTypeLR yLR("values", N, Blk);

    XYTypeLL xLL("values", N, Blk);
    XYTypeLL yLL("values", N, Blk);

    double *s_a = new double[N];
    double *s_b = new double[N];

    if (layout_left)
      printf(" :::: Testing left layout (team_size = %d)\n", team_size);
    if (layout_right)
      printf(" :::: Testing right layout (team_size = %d)\n", team_size);

    if (layout_left) {
      readCRSFromMM(name_A, valuesLL, rowOffsets, colIndices);
      readArrayFromMM(name_B, xLL);
    }
    if (layout_right) {
      readCRSFromMM(name_A, valuesLR, rowOffsets, colIndices);
      readArrayFromMM(name_B, xLR);
    }

    auto alphaV_h = Kokkos::create_mirror_view(alphaV);
    auto betaV_h = Kokkos::create_mirror_view(betaV);

    for (int i = 0; i < N; ++i) {
      s_a[i] = 1.;
      s_b[i] = 0.;
      alphaV_h(i) = s_a[i];
      betaV_h(i) = s_b[i];
    }

    Kokkos::deep_copy(alphaV, alphaV_h);
    Kokkos::deep_copy(betaV, betaV_h);

    using ScratchPadIntView =
        Kokkos::View<int *, exec_space::scratch_memory_space>;

    for (auto i_impl : impls) {
      std::vector<double> timers;

      int n_skip = 2;

      for (int i_rep = 0; i_rep < n_rep_1 + n_skip; ++i_rep) {
        double t_spmv = 0;
        for (int j_rep = 0; j_rep < n_rep_2; ++j_rep) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStart();
#endif
          exec_space().fence();
          if (n_rep_2 != 1)
            flush.run();
          exec_space().fence();

          timer.reset();
          exec_space().fence();

          int N_team = N_team_potential;
          int number_of_teams = ceil(static_cast<double>(N) / N_team);

          if (layout_left) {
            using policy_type = Kokkos::TeamPolicy<exec_space>;
            policy_type auto_policy(number_of_teams, Kokkos::AUTO(),
                                    Kokkos::AUTO());
            policy_type tuned_policy(number_of_teams, team_size,
                                     Kokkos::AUTO());
            policy_type tuned_policy_2(number_of_teams, team_size,
                                       vector_length);
            policy_type policy;

            if (team_size < 1)
              policy = auto_policy;
            else if (vector_length < 1)
              policy = tuned_policy;
            else
              policy = tuned_policy_2;

            // std::cout << "auto_policy.team_size() = " <<
            // auto_policy.team_size() << std::endl;

            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<
                  policy_type, AMatrixValueViewLL, IntView, XYTypeLL, XYTypeLL,
                  alphaViewType, alphaViewType, 0>(policy, alphaV, valuesLL,
                                                   rowOffsets, colIndices, xLL,
                                                   betaV, yLL, N_team)
                  .run();
            } else {
              Kokkos::parallel_for(
                  "KokkosSparse::PerfTest::BSpMV", policy,
                  BSPMV_Functor_View<AMatrixValueViewLL, IntView, XYTypeLL,
                                     XYTypeLL, 0>(s_a, valuesLL, rowOffsets,
                                                  colIndices, xLL, s_b, yLL,
                                                  N_team, N, i_impl));
            }
          }
          if (layout_right) {
            using policy_type = Kokkos::TeamPolicy<exec_space>;
            policy_type auto_policy(number_of_teams, Kokkos::AUTO(),
                                    Kokkos::AUTO());
            policy_type tuned_policy(number_of_teams, team_size,
                                     Kokkos::AUTO());
            policy_type tuned_policy_2(number_of_teams, team_size,
                                       vector_length);
            policy_type policy;

            if (team_size < 1)
              policy = auto_policy;
            else if (vector_length < 1)
              policy = tuned_policy;
            else
              policy = tuned_policy_2;

            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<
                  policy_type, AMatrixValueViewLR, IntView, XYTypeLR, XYTypeLR,
                  alphaViewType, alphaViewType, 0>(policy, alphaV, valuesLR,
                                                   rowOffsets, colIndices, xLR,
                                                   betaV, yLR, N_team)
                  .run();
            } else {
              Kokkos::parallel_for(
                  "KokkosSparse::PerfTest::BSpMV", policy,
                  BSPMV_Functor_View<AMatrixValueViewLR, IntView, XYTypeLR,
                                     XYTypeLR, 0>(s_a, valuesLR, rowOffsets,
                                                  colIndices, xLR, s_b, yLR,
                                                  N_team, N, i_impl));
            }
          }
          exec_space().fence();
          t_spmv += timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStop();
#endif
        }
        if (i_rep > n_skip)
          timers.push_back(t_spmv / n_rep_2);
      }

      {
        std::ofstream myfile;
        std::string name;
        if (layout_left)
          name = name_timer + "_" + std::to_string(i_impl) + "_left.txt";
        if (layout_right)
          name = name_timer + "_" + std::to_string(i_impl) + "_right.txt";

        myfile.open(name);

        for (size_t i = 0; i < timers.size(); ++i)
          myfile << timers[i] << " ";

        myfile << std::endl;

        myfile.close();
      }

      double average_time = 0.;

      for (size_t i = 0; i < timers.size(); ++i)
        average_time += timers[i] / timers.size();

      if (layout_left)
        printf(
            "Left layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, average_time, 1.0 / average_time * 60 * N);
      if (layout_right)
        printf(
            "Right layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, average_time, 1.0 / average_time * 60 * N);

      if (layout_left) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_l.mm", xLL);
      }
      if (layout_right) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_r.mm", xLR);
      }
    }
    */
  }
  Kokkos::finalize();

  return 0;
}
