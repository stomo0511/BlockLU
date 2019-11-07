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

// Show matrix elements
void Show_mat(const int m, const int n, double *A)
{
	for (int i=0; i<m; i++)
	{
		cout << i << ": ";
		for (int j=0; j<n; j++)
			cout << A[i+j*m] << ", ";
		cout << endl;
	}
}

// Debug mode
#define DEBUG

// Trace mode
//#define TRACE

#ifdef TRACE
extern void trace_cpu_start();
extern void trace_cpu_stop(const char *color);
extern void trace_label(const char *color, const char *label);
#endif

int main(const int argc, const char **argv)
{
	// Usage "a.out [size of matrix: m n ] [block width]"
	assert(argc > 3);

	const int m = atoi(argv[1]);     // # rows
	const int n = atoi(argv[2]);     // # columns
	const int nb = atoi(argv[3]);    // Block size
	assert(m >= n);
	assert(nb <= n);

	double *A = new double[m*n];   // Original matrix
	int *piv = new int[m];           // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	cblas_dcopy(m*n,A,1,OA,1);
	double *U = new double[n*n];
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();

	for (int i=0; i<n; i+=nb)
//	for (int i=0; i<nb; i+=nb)
	{
		int ib = min(n-i,nb);

		{
			#ifdef TRACE
			trace_cpu_start();
			trace_label("Red", "dgetrf2");
			#endif

			assert(0 == LAPACKE_dgetrf2(MKL_COL_MAJOR, m-i, ib, A+(i+i*m), m, piv+i));

			for (int k=i; k<min(m,i+ib); k++)
				piv[k] += i;

			#ifdef TRACE
			trace_cpu_stop("Red");
			#endif
		}

		{
			#ifdef TRACE
			trace_cpu_start();
			trace_label("Aqua", "dlaswp1");
			#endif

			// Apply interchanges to columns 0:i
			assert(0 == LAPACKE_dlaswp(MKL_COL_MAJOR, i, A, m, i+1, i+ib, piv, 1));

			#ifdef TRACE
			trace_cpu_stop("Aqua");
			#endif
		}

		if (i+ib < n)
		{
			#pragma omp parallel for
			for (int j=i+ib; j<n; j+=ib)
			{
				int jb = min(n-j,nb);

				{
					#ifdef TRACE
					trace_cpu_start();
					trace_label("LightCyan", "dlaswp2");
					#endif

					// Apply interchanges to columns i+ib:n-1
					assert(0 == LAPACKE_dlaswp(MKL_COL_MAJOR, jb, A+(j*m), m, i+1, i+ib, piv, 1));

					#ifdef TRACE
					trace_cpu_stop("LightCyan");
					#endif
				}

				{
					#ifdef TRACE
					trace_cpu_start();
					trace_label("Green", "dtrsm");
					#endif

					// Compute block row of U
					cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
							ib, jb, 1.0, A+(i+i*m), m, A+(i+j*m), m);

					#ifdef TRACE
					trace_cpu_stop("Green");
					#endif
				}

				// Update trailing submatrix
				if (i+ib < m) {
					#ifdef TRACE
					trace_cpu_start();
					trace_label("Blue", "dgemm");
					#endif

					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							m-i-ib, jb, ib, -1.0, A+(i+ib+i*m), m, A+(i+j*m), m, 1.0, A+(i+ib+j*m), m);

					#ifdef TRACE
					trace_cpu_stop("Blue");
					#endif
				}
			} // End of j-loop
		} // End of if
	} // End of i-loop

	timer = omp_get_wtime() - timer;

	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;

	Show_mat(m,n,A);

	////////// Debug mode //////////
	#ifdef DEBUG
	// Upper triangular matrix
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++)
			U[i+j*n] = (j<i) ? 0.0 : A[i+j*m];

	// Unit lower triangular matrix
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			if (i==j)
				A[i+j*m] = 1.0;
			else if (j>i)
				A[i+j*m] = 0.0;
		}

	// Apply interchanges to original matrix A
	assert(0 == LAPACKE_dlaswp(MKL_COL_MAJOR, n, OA, m, 1, n, piv, 1));

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			m, n, n, -1.0, A, m, U, n, 1.0, OA, m);

	cout << "Debug mode: \n";
	cout << "|| PA - LU ||_2 = " << cblas_dnrm2(m*n, OA, 1) << endl;

	delete [] OA;
	delete [] U;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] piv;

	return EXIT_SUCCESS;
}
