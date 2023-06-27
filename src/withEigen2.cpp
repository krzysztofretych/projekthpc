#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <benchmark/benchmark.h>

typedef Eigen::SparseMatrix<double> SpMat; // Sparse matrix type
typedef Eigen::Triplet<double> Triplet; // Sparse matrix coefficients

static void BM_NavierStokes(benchmark::State& state) {
    // Define the grid size
    int N = 100; // Number of grid points

    // Define the grid spacing
    double h = 1.0 / (N - 1);

    // Set the simulation parameters
    double dt = 0.001; // Time step size
    double T = 1.0; // Total simulation time

    // Set the fluid properties
    double nu = 0.1; // Viscosity
    double rho = 1.0; // Density

    // Calculate the number of time steps
    int numSteps = static_cast<int>(T / dt);

    // Define the velocity field
    Eigen::VectorXd u(N * N);
    Eigen::VectorXd v(N * N);

    // Define the pressure field
    Eigen::VectorXd p(N * N);

    // Define the right-hand side vector
    Eigen::VectorXd rhs(N * N);

    // Define the sparse matrix coefficients
    std::vector<Triplet> coeffs;
    coeffs.reserve(5 * N * N);

    // Construct the sparse matrix
    SpMat A(N * N, N * N);

    // Set up the initial conditions
    u.setZero();
    v.setZero();
    p.setZero();

    // Time integration loop
    //const int size = state.range(0);
    for (auto _ : state) {
        for (int step = 0; step < numSteps; ++step) {
            // Calculate the right-hand side vector
            rhs.setZero();
            printf("Right hand side vector calculated.");

            // Assemble the coefficients for the sparse matrix
            coeffs.clear();
            printf("Coefficients for the sparse matrix assembled.");

            // Construct the sparse matrix
            A.setZero();
            printf("Sparse matrix constructed");

            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int index = i * N + j;

                    // Set up the coefficients for the Navier-Stokes equations
                    coeffs.push_back(Triplet(index, index, 1.0 + 4.0 * nu * dt / (h * h)));

                    coeffs.push_back(Triplet(index, index - 1, -nu * dt / (h * h)));
                    coeffs.push_back(Triplet(index, index + 1, -nu * dt / (h * h)));
                    coeffs.push_back(Triplet(index, index - N, -nu * dt / (h * h)));
                    coeffs.push_back(Triplet(index, index + N, -nu * dt / (h * h)));

                    // Calculate the right-hand side vector
                    rhs(index) = u(index) - dt / rho * (p(index + N) - p(index)) / h;
                }
            }

            // Apply the boundary conditions

            // Bottom wall (no-slip condition)
            for (int j = 0; j < N; ++j) {
                int index = j;
                u(index) = 0.0;
                v(index) = 0.0;
            }

            // Top wall (lid-driven condition)
            for (int j = 0; j < N; ++j) {
                int index = (N - 1) * N + j;
                u(index) = 1.0;
                v(index) = 0.0;
            }

            // Left wall (no-slip condition)
            for (int i = 1; i < N - 1; ++i) {
                int index = i * N;
                u(index) = 0.0;
                v(index) = 0.0;
            }

            // Right wall (no-slip condition)
            for (int i = 1; i < N - 1; ++i) {
                int index = i * N + N - 1;
                u(index) = 0.0;
                v(index) = 0.0;
            }
            printf("Boundary conditions applied.");

            // Solve the sparse linear system
            A.setFromTriplets(coeffs.begin(), coeffs.end());

            Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> solver;
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Error: Failed to decompose the matrix." << std::endl;
                std::cerr << "Solver error: " << solver.lastErrorMessage() << std::endl;
                return;
            }

            u = solver.solve(rhs);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Error: Failed to solve the linear system." << std::endl;
                std::cerr << "Solver error: " << solver.lastErrorMessage() << std::endl;
                return;
            }

            // Update the pressure field
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int index = i * N + j;
                    p(index) -= dt * rho * h * h / (4.0 * nu) * ((u(index + 1) - u(index - 1)) / (2.0 * h) + (v(index + N) - v(index - N)) / (2.0 * h));
                }
            }
            printf("Pressure field updated.");

            // Update the velocity field
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int index = i * N + j;
                    u(index) -= dt / (rho * h) * (p(index + N) - p(index));
                    v(index) -= dt / (rho * h) * (p(index + 1) - p(index));
                }
            }
            printf("Velocity field updated.");
        }
    }

    // Print the final results
    std::cout << "Final velocity field:\n" << u << std::endl;
    std::cout << "Final pressure field:\n" << p << std::endl;
}

BENCHMARK(BM_NavierStokes);

BENCHMARK_MAIN();
