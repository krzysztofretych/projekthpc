#include <iostream>
 //#include <Eigen/Core>
#include <Eigen/Dense>

#include <benchmark/benchmark.h>

//#define grid 128
const static int grid = 128;

static void BM_NavierStokes(benchmark::State & state) {
  
  const int size = state.range(0);
  for (auto _: state) {
    Eigen::MatrixXd u(grid, grid + 1), un(grid, grid + 1), uc(grid, grid);
    Eigen::MatrixXd v(grid + 1, grid), vn(grid + 1, grid), vc(grid, grid);
    Eigen::MatrixXd p(grid + 1, grid + 1), pn(grid + 1, grid + 1), pc(grid, grid);
    Eigen::MatrixXd m(grid + 1, grid + 1);
    int i, j, step;
    double dx, dy, dt, tau, delta, error, Re;
    step = 1;
    dx = 1.0 / (grid - 1);
    dy = 1.0 / (grid - 1);
    dt = 0.001;
    delta = 4.5;
    error = 1.0;
    Re = 100.0;

    // Initializing u
    u.setZero();
    u.col(grid) = Eigen::VectorXd::Constant(grid, 1.0);
    u.col(grid - 1) = Eigen::VectorXd::Constant(grid, 1.0);

    // Initializing v
    v.setZero();

    // Initializing p
    p.setOnes();

    while (error > 0.00001) {
      // Solve u-momentum equation
      un.block(1, 1, grid - 2, grid - 1) =
        u.block(1, 1, grid - 2, grid - 1).array() -
        dt * ((u.block(2, 1, grid - 2, grid - 1).array() * u.block(2, 1, grid - 2, grid - 1).array() -
            u.block(0, 1, grid - 2, grid - 1).array() * u.block(0, 1, grid - 2, grid - 1).array()) *
          0.5 * 1.0 / dx +
          0.25 * ((u.block(1, 1, grid - 2, grid - 1).array() + u.block(1, 2, grid - 2, grid - 1).array()) *
            (v.block(1, 1, grid - 2, grid - 1).array() + v.block(2, 1, grid - 2, grid - 1).array()) -
            (u.block(1, 1, grid - 2, grid - 1).array() + u.block(1, 0, grid - 2, grid - 1).array()) *
            (v.block(2, 0, grid - 2, grid - 1).array() + v.block(1, 0, grid - 2, grid - 1).array())) *
          1.0 / dy) -
        dt / dx * (p.block(2, 1, grid - 2, grid - 1).array() - p.block(1, 1, grid - 2, grid - 1).array()) +
        dt * 1.0 / Re *
        ((u.block(2, 1, grid - 2, grid - 1).array() - 2.0 * u.block(1, 1, grid - 2, grid - 1).array() +
            u.block(0, 1, grid - 2, grid - 1).array()) /
          (dx * dx) +
          (u.block(1, 2, grid - 2, grid - 1).array() - 2.0 * u.block(1, 1, grid - 2, grid - 1).array() +
            u.block(1, 0, grid - 2, grid - 1).array()) /
          (dy * dy));

      // Boundary conditions
      un.row(0).setZero();
      un.row(grid - 1).setZero();
      un.col(0) = -un.col(1);
      un.col(grid).array() = - un.col(grid - 1).array() + 2.0;

      // Solve v-momentum
      vn.block(1, 1, grid - 1, grid - 2) =
        v.block(1, 1, grid - 1, grid - 2).array() -
        dt * (0.25 * ((u.block(1, 1, grid - 1, grid - 2).array() + u.block(1, 2, grid - 1, grid - 2).array()) *
            (v.block(1, 1, grid - 1, grid - 2).array() + v.block(2, 1, grid - 1, grid - 2).array()) -
            (u.block(0, 1, grid - 1, grid - 2).array() + u.block(0, 2, grid - 1, grid - 2).array()) *
            (v.block(1, 1, grid - 1, grid - 2).array() + v.block(0, 1, grid - 1, grid - 2).array())) /
          dx +
          (v.block(1, 2, grid - 1, grid - 2).array() * v.block(1, 2, grid - 1, grid - 2).array() -
            v.block(1, 0, grid - 1, grid - 2).array() * v.block(1, 0, grid - 1, grid - 2).array()) /
          (2.0 * dy)) -
        dt / dy * (p.block(1, 2, grid - 1, grid - 2).array() - p.block(1, 1, grid - 1, grid - 2).array()) +
        dt * 1.0 / Re *
        ((v.block(2, 1, grid - 1, grid - 2).array() - 2.0 * v.block(1, 1, grid - 1, grid - 2).array() +
            v.block(0, 1, grid - 1, grid - 2).array()) /
          (dx * dx) +
          (v.block(1, 2, grid - 1, grid - 2).array() - 2.0 * v.block(1, 1, grid - 1, grid - 2).array() +
            v.block(1, 0, grid - 1, grid - 2).array()) /
          (dy * dy));

      // Boundary conditions
      vn.row(0) = -vn.row(1);
      vn.row(grid) = -vn.row(grid - 1);
      vn.col(0).setZero();
      vn.col(grid - 1).setZero();

      // Solves continuity equation
      pn.block(1, 1, grid - 1, grid - 1) =
        p.block(1, 1, grid - 1, grid - 1).array() -
        dt * delta * ((un.block(1, 1, grid - 1, grid - 1).array() - un.block(0, 1, grid - 1, grid - 1).array()) * 1.0 / dx +
          (vn.block(1, 1, grid - 1, grid - 1).array() - vn.block(1, 0, grid - 1, grid - 1).array()) / dy);

      // Boundary conditions
      pn.block(1, 0, grid - 1, 1) = pn.block(1, 1, grid - 1, 1);
      pn.block(1, grid, grid - 1, 1) = pn.block(1, grid - 1, grid - 1, 1);
      pn.block(0, 1, 1, grid - 1) = pn.block(1, 1, 1, grid - 1);
      pn.block(grid, 1, 1, grid - 1) = pn.block(grid - 1, 1, 1, grid - 1);

      // Displaying error
      error = (pn.block(1, 1, grid - 1, grid - 1) - p.block(1, 1, grid - 1, grid - 1)).array().abs().sum();

      if (step%1000 ==1)
		{
	        printf("Error is %5.8lf for the step %d\n", error, step);
		}

      // Iterating u
      u.block(0, 0, grid, grid + 1) = un.block(0, 0, grid, grid + 1);

      // Iterating v
      v.block(0, 0, grid + 1, grid) = vn.block(0, 0, grid + 1, grid);

      // Iterating p
      p.block(0, 0, grid + 1, grid + 1) = pn.block(0, 0, grid + 1, grid + 1);

      step++;
    }

    for (int i = 0; i < grid - 1; i++) {
      for (int j = 0; j < grid - 1; j++) {
        uc(i, j) = 0.5 * (u(i, j) + u(i, j + 1));
        vc(i, j) = 0.5 * (v(i, j) + v(i + 1, j));
        pc(i, j) = 0.25 * (p(i, j) + p(i + 1, j) + p(i, j + 1) + p(i + 1, j + 1));
      }
    }
  }
}

BENCHMARK(BM_NavierStokes) -> UseRealTime() -> Unit(benchmark::kMillisecond) -> Ranges({{128, 2 << 9},{1, 4}});

BENCHMARK_MAIN();