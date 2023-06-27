#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <benchmark/benchmark.h>

#define grid 128

using namespace Eigen;

static void BM_NavierStokes(benchmark::State& state)
//int main(void)
{
    const int size = state.range(0);
    for (auto _ : state)
    {
        SparseMatrix<double> u(grid, grid + 1);
        SparseMatrix<double> un(grid, grid + 1);
        MatrixXd uc(grid, grid);

        SparseMatrix<double> v(grid + 1, grid);
        SparseMatrix<double> vn(grid + 1, grid);
        MatrixXd vc(grid, grid);

        MatrixXd p(grid + 1, grid + 1);
        MatrixXd pn(grid + 1, grid + 1);
        MatrixXd pc(grid, grid);

        MatrixXd m(grid + 1, grid + 1);

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
        for (i = 0; i <= (grid - 1); i++)
        {
            u.coeffRef(i, grid) = 1.0;
            u.coeffRef(i, grid - 1) = 1.0;
        }

        // Initializing v
        v.setZero();

        // Initializing p
        p.setOnes();
        /* for (int k = 0; k < p.nonZeros(); ++k) {
            p.valuePtr()[k] = 1.0;
        } */

        while (error > 0.00001)
        {
            // Solve u-momentum equation
            for (i = 1; i <= (grid - 2); i++)
            {
                for (j = 1; j <= (grid - 1); j++)
                {
                    un.coeffRef(i, j) = u.coeff(i, j) - dt * ((u.coeff(i + 1, j) * u.coeff(i + 1, j) - u.coeff(i - 1, j) * u.coeff(i - 1, j)) / 2.0 / dx +
                                                              0.25 * ((u.coeff(i, j) + u.coeff(i, j + 1)) * (v.coeff(i, j) + v.coeff(i + 1, j)) -
                                                                      (u.coeff(i, j) + u.coeff(i, j - 1)) * (v.coeff(i + 1, j - 1) + v.coeff(i, j - 1))) /
                                                                          dy) -
                                       dt / dx * (p.coeff(i + 1, j) - p.coeff(i, j)) +
                                       dt * 1.0 / Re * ((u.coeff(i + 1, j) - 2.0 * u.coeff(i, j) + u.coeff(i - 1, j)) / dx / dx +
                                                        (u.coeff(i, j + 1) - 2.0 * u.coeff(i, j) + u.coeff(i, j - 1)) / dy / dy);
                }
            }

            // Boundary conditions
            for (j = 1; j <= (grid - 1); j++)
            {
                un.coeffRef(0, j) = 0.0;
                un.coeffRef(grid - 1, j) = 0.0;
            }

            for (i = 1; i <= (grid - 1); i++)
            {
                un.coeffRef(i, 0) = -un.coeff(i, 1);
                un.coeffRef(i, grid) = 2.0 - un.coeff(i, grid - 1);
            }

            // Solve v-momentum equation
            for (i = 1; i <= (grid - 1); i++)
            {
                for (j = 1; j <= (grid - 2); j++)
                {
                    vn.coeffRef(i, j) = v.coeff(i, j) - dt * (0.25 * ( (u.coeff(i, j)+u.coeff(i, j+1))*(v.coeff(i, j)+v.coeff(i+1, j))-(u.coeff(i-1, j)+u.coeff(i-1, j+1))*(v.coeff(i, j)+v.coeff(i-1, j)))/dx 
							+(v.coeff(i, j+1)*v.coeff(i, j+1)-v.coeff(i, j-1)*v.coeff(i, j-1))/2.0/dy ) 
							- dt/dy*(p.coeff(i, j+1)-p.coeff(i, j)) 
							+ dt*1.0/Re*((v.coeff(i+1, j)-2.0*v.coeff(i, j)+v.coeff(i-1, j))/dx/dx+(v.coeff(i, j+1)-2.0*v.coeff(i, j)+v.coeff(i, j-1))/dy/dy);
                }
            }

            // Boundary conditions
            for (j = 1; j <= (grid - 2); j++)
            {
                vn.coeffRef(0, j) = -vn.coeff(1, j);
                vn.coeffRef(grid, j) = -vn.coeff(grid - 1, j);
            }

            for (i = 0; i <= (grid); i++)
            {
                vn.coeffRef(i, 0) = 0.0;
                vn.coeffRef(i, grid - 1) = 0.0;
            }

            // Solve continuity equation
            for (i=1; i<=(grid-1); i++)
            {
                for (j=1; j<=(grid-1); j++)
                {
                    pn.coeffRef(i, j) = p.coeff(i, j)-dt*delta*(  ( un.coeff(i, j)-un.coeff(i-1, j) )/dx + ( vn.coeff(i, j)-vn.coeff(i, j-1) ) /dy  );
                }
            }
            
            
            /* // Solve pressure equation
            pc = p;
            p.setZero();

            tau = 0.05;
            int iter = 0;
            error = 1.0;
            while (error > 0.00000001)
            {
                pn = p;

                for (i = 1; i <= (grid - 1); i++)
                {
                    for (j = 1; j <= (grid - 1); j++)
                    {
                        p.coeffRef(i, j) = (1.0 - tau) * pn.coeff(i, j) +
                                           tau / (2.0 * dx * dx + 2.0 * dy * dy) *
                                               ((pn.coeff(i + 1, j) + pn.coeff(i - 1, j)) * dy * dy +
                                                (pn.coeff(i, j + 1) + pn.coeff(i, j - 1)) * dx * dx -
                                                dx * dx * dy * dy * m.coeff(i, j));
                    }
                } */

            // Boundary conditions
            for (i = 1; i <= (grid - 1); i++)
            {
                pn.coeffRef(i, 0) = pn.coeff(i, 1);
                pn.coeffRef(i, grid) = pn.coeff(i, grid - 1);
            }

            for (j = 0; j <= grid; j++)
            {
                pn.coeffRef(0, j) = pn.coeff(1, j);
                pn.coeffRef(grid, j) = pn.coeff(grid - 1, j);
            }

            // Displaying error
            error = 0.0;
		
            for (i=1; i<=(grid-1); i++)
            {
                for (j=1; j<=(grid-1); j++)
                {
                    m.coeffRef(i, j) = (  ( un.coeff(i, j)-un.coeff(i-1, j) )/dx + ( vn.coeff(i, j)-vn.coeff(i, j-1) )/dy  );
                    error = error + fabs(m.coeff(i, j));
                }
            }
		
            if (step%1000 == 1)
            {
                printf("Error is %5.8lf for the step %d\n", error, step);
            }
            
            /* // Error calculation
            error = (p - pn).norm() / p.norm();
            iter++;
            

            std::cout << "Iteration: " << iter << ", Error: " << error << std::endl;
 */
            // Update velocities
            /* uc = u;
            vc = v; */

            /* for (i = 1; i <= (grid - 2); i++)
            {
                for (j = 1; j <= (grid - 1); j++)
                {
                    u.coeffRef(i, j) = un.coeff(i, j) - dt / dx * (p.coeff(i + 1, j) - p.coeff(i, j));
                }
            } */

            /* for (i = 1; i <= (grid - 1); i++)
            {
                for (j = 1; j <= (grid - 2); j++)
                {
                    v.coeffRef(i, j) = vn.coeff(i, j) - dt / dy * (p.coeff(i, j + 1) - p.coeff(i, j));
                }
            } */

            /* // Boundary conditions
            for (j = 1; j <= (grid - 1); j++)
            {
                u.coeffRef(0, j) = 0.0;
                u.coeffRef(grid - 1, j) = 0.0;
            } */

            /* for (i = 1; i <= (grid - 1); i++)
            {
                v.coeffRef(i, 0) = 0.0;
                v.coeffRef(i, grid - 1) = 0.0;
            } */

            // Calculate mass conservation error
            /* for (i = 1; i <= (grid - 2); i++)
            {
                for (j = 1; j <= (grid - 2); j++)
                {
                    m.coeffRef(i, j) = (uc.coeff(i + 1, j) - uc.coeff(i - 1, j)) / 2.0 / dx +
                                       (vc.coeff(i, j + 1) - vc.coeff(i, j - 1)) / 2.0 / dy;
                }
            }

            double m_error = m.sum();
            std::cout << "Mass Conservation Error: " << m_error << std::endl;

            step++; */

            // Iterating u
            for (i=0; i<=(grid-1); i++)
            {
                for (j=0; j<=(grid); j++)
                {
                    u.coeffRef(i, j) = u.coeff(i, j);
                }
            }
		
            // Iterating v
            for (i=0; i<=(grid); i++)
            {
                for (j=0; j<=(grid-1); j++)
                {
                    v.coeffRef(i, j) = vn.coeff(i, j);
                }
            }
		
            // Iterating p
            for (i=0; i<=(grid); i++)
            {
                for (j=0; j<=(grid); j++)
                {
                    p.coeffRef(i, j) = pn.coeff(i, j);
                }
            }

            step++;

        }
	
        for (i=0; i<=(grid-1); i++)
        {
            for (j=0; j<=(grid-1); j++)
            {
                uc.coeffRef(i, j) = 0.5*(u.coeff(i, j)+u.coeff(i, j+1));
                vc.coeffRef(i, j) = 0.5*(v.coeff(i, j)+v.coeff(i+1, j));
                pc.coeffRef(i, j) = 0.25*(p.coeff(i, j)+p.coeff(i+1, j)+p.coeff(i, j+1)+p.coeff(i+1, j+1));
            }
        }
        
    }
}

/* 	// OUTPUT DATA
    //FILE fout2, fout3;
	FILE *fout2, *fout3;
	fout2 = fopen("UVP_Eigen.plt","w+t");
	fout3 = fopen("Central_U_Eigen.plt","w+t");

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
    } */

BENCHMARK(BM_NavierStokes)->UseRealTime()->Ranges({{128, 2<<9}, {1, 4}});

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
