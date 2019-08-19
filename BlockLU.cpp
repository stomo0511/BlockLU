//============================================================================
// Name        : BlockLU.cpp
// Author      : Tomo Suzuki
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <mkl.h>

using namespace std;

// Generate random matrix
void Gen_rand_mat(const int m, const int n, double *A)
{
	srand(20190726);

//	#pragma omp parallel for
	for (int i=0; i<m*n; i++)
		A[i] = 1.0 - 2*(double)rand() / RAND_MAX;
}

// Show matrix
void Show_mat(const int m, const int n, double *A)
{
	cout.setf(ios::scientific);
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			cout << showpos << setprecision(4) << A[i + j*m] << ", ";
		cout << endl;
	}
	cout << endl;
}

// Debug mode
#define DEBUG

int main(const int argc, const char **argv)
{
	// Usage "a.out [size of matrix: m n ] [block width]"
	assert(argc > 3);

	const int m = atoi(argv[1]);     // # rows
	const int n = atoi(argv[2]);     // # columns
	const int nb = atoi(argv[3]);    // Block size
	assert(m >= n);
	assert(nb <= n);

	double *A = new double[m*n];     // Original matrix
	int *piv = new int[m];           // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	cblas_dcopy(m*n,A,1,OA,1);
	double *U = new double[m*n];
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();

	#pragma omp parallel
	{
		#pragma omp single
		{
			for (int i=0; i<n; i+=nb)
			{
				int ib = min(n-i,nb);

				#pragma omp task depend(inout: A[i*m:m*ib]) depend(out: piv[i:ib])
				{
//					trace_cpu_start();
//					trace_label("Red", "GETRF2");

					int info = LAPACKE_dgetrf2(MKL_COL_MAJOR, m-i, ib, A+(i+i*m), m, piv+i);
					assert(info==0);

					for (int k=i; k<min(m,i+ib); k++)
						piv[k] += i;

//					trace_cpu_stop("Red");
				}

				#pragma omp task depend(inout: A[0:m*i]) depend(in: piv[i:ib])
				{
//					trace_cpu_start();
//					trace_label("Aqua", "LASWP1");

					// Apply interchanges to columns 0:i
					int info = LAPACKE_dlaswp(MKL_COL_MAJOR, i, A, m, i+1, i+ib, piv, 1);
					assert(info==0);

//					trace_cpu_stop("Aqua");
				}

				if (i+ib < n)
				{
					for (int j=i+ib; j<n; j+=ib)
					{
						int jb = min(n-j,nb);

						#pragma omp task depend(inout: A[j*m:m*jb]) depend(in: piv[i:ib])
						{
//							trace_cpu_start();
//							trace_label("LightCyan", "LASWP2");

							// Apply interchanges to columns i+ib:n-1
							int info = LAPACKE_dlaswp(MKL_COL_MAJOR, jb, A+(j*m), m, i+1, i+ib, piv, 1);
							assert(info==0);

//							trace_cpu_stop("LightCyan");
						}

						#pragma omp task depend(in: A[i*m:m*ib]) depend(inout: A[j*m:m*jb])
						{
//							trace_cpu_start();
//							trace_label("Green", "TRSM");

							// Compute block row of U
							cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
									ib, jb, 1.0, A+(i+i*m), m, A+(i+j*m), m);

//							trace_cpu_stop("Green");
						}

						// Update trailing submatrix
						if (i+ib < m) {
						#pragma omp task depend(in: A[i*m:m*ib]) depend(inout: A[j*m:m*jb])
						{
//							trace_cpu_start();
//							trace_label("Blue", "GEMM");

							cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
									m-i-ib, jb, ib, -1.0, A+(i+ib+i*m), m, A+(i+j*m), m, 1.0, A+(i+ib+j*m), m);

//							trace_cpu_stop("Blue");
						} }
					} // End of j-loop
				} // End of if
			} // End of i-loop
		} // End of Single
	} // End of Parallel

	timer = omp_get_wtime() - timer;

	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;

	////////// Debug mode //////////
	#ifdef DEBUG
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			U[i+j*m] = (j<i) ? 0.0 : A[i+j*m];

	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
			m, n, 1.0, A, m, U, m);

	// Apply interchanges to matrix A
	assert( 0 == LAPACKE_dlaswp(MKL_COL_MAJOR, n, OA, m, 1, n, piv, 1));

	double tmp = 0.0;
	for (int i=0; i<m*n; i++)
		tmp += (OA[i] - U[i])*(OA[i] - U[i]);

	cout << "Debug mode: \n";
	cout << "|| PA - LU ||_2 = " << sqrt(tmp) << endl;

	delete [] OA;
	delete [] U;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] piv;

	return EXIT_SUCCESS;
}
