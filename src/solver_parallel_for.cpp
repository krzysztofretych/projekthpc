#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <benchmark/benchmark.h>
#include <tbb/tbb.h>
#include <tbb/global_control.h>
#define grid 128

//static void BM_NavierStokes(benchmark::State& state)
int main (void)
{
    //const int size = state.range(0);
	//int no_threads = state.range(1);
	//tbb::global_control c(tbb::global_control::max_allowed_parallelism, no_threads);
    //for (auto _ : state)
    //{
        double u[grid][grid+1], un[grid][grid+1], uc[grid][grid];
        double v[grid+1][grid], vn[grid+1][grid], vc[grid][grid];
        double p[grid+1][grid+1], pn[grid+1][grid+1], pc[grid][grid];
        double m[grid+1][grid+1];
        int i, j, step;
        double dx, dy, dt, tau, delta, error, Re;
        step = 1;
        dx = 1.0/(grid-1);
        dy = 1.0/(grid-1);
        dt = 0.001;
        delta = 4.5;
        error = 1.0;
        Re = 100.0;

        // Initializing u
        tbb::parallel_for(tbb::blocked_range<int>(0, grid), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                for (int j = 0; j <= grid; j++) {
                    u[i][j] = 0.0;
                    u[i][grid] = 1.0;
                    u[i][grid-1] = 1.0;
                }
            }
        });

        // Initializing v
        tbb::parallel_for(tbb::blocked_range<int>(0, grid + 1), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                for (int j = 0; j <= grid - 1; j++) {
                    v[i][j] = 0.0;
                }
            }
        });

        // Initializing p
        tbb::parallel_for(tbb::blocked_range<int>(0, grid + 1), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                for (int j = 0; j <= grid; j++) {
                    p[i][j] = 1.0;
                }
            }
        });

        while (error > 0.00001)
        {
            // Solve u-momentum equation
            tbb::parallel_for(tbb::blocked_range<int>(1, grid - 1), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    for (int j = 1; j <= grid - 1; j++) {
                        un[i][j] = u[i][j] - dt * ((u[i + 1][j] * u[i + 1][j] - u[i - 1][j] * u[i - 1][j]) / 2.0 / dx
                            + 0.25 * ((u[i][j] + u[i][j + 1]) * (v[i][j] + v[i + 1][j]) - (u[i][j] + u[i][j - 1]) * (v[i + 1][j - 1] + v[i][j - 1])) / dy)
                            - dt / dx * (p[i + 1][j] - p[i][j])
                            + dt * 1.0 / Re * ((u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / dx / dx + (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / dy / dy);
                    }
                }
            });

            // Boundary conditions
            tbb::parallel_for(tbb::blocked_range<int>(1, grid - 1), [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j != r.end(); ++j) {
                    un[0][j] = 0.0;
                    un[grid - 1][j] = 0.0;
                }
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, grid), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    un[i][0] = -un[i][1];
                    un[i][grid] = 2 - un[i][grid - 1];
                }
            });

            // Solves v-momentum
            tbb::parallel_for(tbb::blocked_range<int>(1, grid), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    for (int j = 1; j <= grid - 2; j++) {
                        vn[i][j] = v[i][j] - dt * (0.25 * ((u[i][j] + u[i][j + 1]) * (v[i][j] + v[i + 1][j]) - (u[i - 1][j] + u[i - 1][j + 1]) * (v[i][j] + v[i - 1][j])) / dx
                            + (v[i][j + 1] * v[i][j + 1] - v[i][j - 1] * v[i][j - 1]) / 2.0 / dy)
                            - dt / dy * (p[i][j + 1] - p[i][j])
                            + dt * 1.0 / Re * ((v[i + 1][j] - 2.0 * v[i][j] + v[i - 1][j]) / dx / dx + (v[i][j + 1] - 2.0 * v[i][j] + v[i][j - 1]) / dy / dy);
                    }
                }
            });

            // Boundary conditions
            tbb::parallel_for(tbb::blocked_range<int>(1, grid - 2), [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j != r.end(); ++j) {
                    vn[0][j] = -vn[1][j];
                    vn[grid][j] = -vn[grid - 1][j];
                }
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, grid + 1), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    vn[i][0] = 0.0;
                    vn[i][grid - 1] = 0.0;
                }
            });

            // Solves continuity equation
            tbb::parallel_for(tbb::blocked_range<int>(1, grid), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    for (int j = 1; j <= grid - 1; j++) {
                        pn[i][j] = p[i][j] - dt * delta * ((un[i][j] - un[i - 1][j]) / dx + (vn[i][j] - vn[i][j - 1]) / dy);
                    }
                }
            });

            // Boundary conditions
            tbb::parallel_for(tbb::blocked_range<int>(1, grid - 1), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    pn[i][0] = pn[i][1];
                    pn[i][grid] = pn[i][grid - 1];
                }
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, grid + 1), [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j != r.end(); ++j) {
                    pn[0][j] = pn[1][j];
                    pn[grid][j] = pn[grid - 1][j];
                }
            });

            // Displaying error
            error = 0.0;
            tbb::parallel_for(tbb::blocked_range2d<int>(1, grid - 1, 1, grid - 1), [&](const tbb::blocked_range2d<int>& r) {
                for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                    for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                        m[i][j] = ((un[i][j] - un[i - 1][j]) / dx + (vn[i][j] - vn[i][j - 1]) / dy);
                        error += fabs(m[i][j]);
                    }
                }
            });

            if (step % 1000 == 1) {
                printf("Error is %5.8lf for the step %d\n", error, step);
            }

            // Iterating u
            tbb::parallel_for(tbb::blocked_range2d<int>(0, grid - 1, 0, grid), [&](const tbb::blocked_range2d<int>& r) {
                for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                    for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                        u[i][j] = un[i][j];
                    }
                }
            });

            // Iterating v
            tbb::parallel_for(tbb::blocked_range2d<int>(0, grid, 0, grid - 1), [&](const tbb::blocked_range2d<int>& r) {
                for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                    for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                        v[i][j] = vn[i][j];
                    }
                }
            });

            // Iterating p
            tbb::parallel_for(tbb::blocked_range2d<int>(0, grid + 1, 0, grid + 1), [&](const tbb::blocked_range2d<int>& r) {
                for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                    for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                        p[i][j] = pn[i][j];
                    }
                }
            });

            step++;
        }
    //}
 	
	//OUTPUT DATA
	FILE *fout2, *fout3;
	fout2 = fopen("UVP_parallel.plt","w+t");
	fout3 = fopen("Central_U_parallel.plt","w+t");

	if ( fout2 == NULL )
	{
    printf("\nERROR when opening file\n");
    fclose( fout2 );
	}

else
	{
	fprintf( fout2, "VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"P\"\n");
	fprintf( fout2, "ZONE  F=POINT\n");
	fprintf( fout2, "I=%d, J=%d\n", grid, grid );

	for ( j = 0 ; j < (grid) ; j++ )
	{
    for ( i = 0 ; i < (grid) ; i++ )
    {
		double xpos, ypos;
		xpos = i*dx;
		ypos = j*dy;

		fprintf( fout2, "%5.8lf\t%5.8lf\t%5.8lf\t%5.8lf\t%5.8lf\n", xpos, ypos, uc[i][j], vc[i][j], pc[i][j] );
    }
	}
	}

	fclose( fout2 );
	
// CENTRAL --U
  fprintf(fout3, "VARIABLES=\"U\",\"Y\"\n");
  fprintf(fout3, "ZONE F=POINT\n");
  fprintf(fout3, "I=%d\n", grid );

  for ( j = 0 ; j < grid ; j++ )
  {
	double ypos;
    ypos = (double) j*dy;

    fprintf( fout3, "%5.8lf\t%5.8lf\n", (uc[grid/2][j] + uc[(grid/2)+1][j])/(2.), ypos ); 
  }
}

//BENCHMARK(BM_NavierStokes)->UseRealTime()->Ranges({{128, 2<<9}, {1, 4}});;

//BENCHMARK_MAIN();
