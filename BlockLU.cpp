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

// Copy matrix elements from A to B
void Copy_mat(const int m, const int n, double *A, double *B)
{
//	#pragma omp for
	for (int i=0; i<m*n; i++)
		B[i] = A[i];
}

// my_dgetrf2
void My_dgetrf2(MKL_INT M, MKL_INT N, double* A, MKL_INT lda, int* PIV, int* INFO)
{
	int inf;
	dgetrf2_( &M, &N, A, &lda, PIV, &inf);
	*INFO = inf;
}

// my_dlaswp
void My_dlaswp(MKL_INT N, double* A, MKL_INT lda, MKL_INT K1, MKL_INT K2, int* PIV, MKL_INT INCX)
{
	dlaswp_( &N, A, &lda, &K1, &K2, PIV, &INCX);
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

	double *A = new double [m*n];  // Original matrix
	int *piv = new int [m];          // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix

	// Dependency checker
	const int p = (m % nb == 0) ? m / nb : m / nb +1;
	const int q = (n % nb == 0) ? n / nb : n / nb +1;
	int **dep = new int* [p];
	for (int i=0; i<p; i++)
		dep[i] = new int [q];

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	Copy_mat(m,n,A,OA);
	double *U = new double[m*n];
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();   // Timer start

	#pragma omp parallel
	{
		#pragma omp single
		{
			for (int i=0; i<n; i+=nb)
			{
				int ib = min(n-i,nb);

				#pragma omp task depend(inout: dep[i/ib:p-i/ib][i/ib]) depend(out: piv[i:ib])
				{
//					#pragma omp critical
//					cout << "dgetrf2: inout: dep[" << i/ib << ":" << p-i/ib << "][" << i/ib << "], out: piv[" << i << ":" << ib << "]\n";

					int info = LAPACKE_dgetrf2(MKL_COL_MAJOR, m-i, ib, A+(i+i*m), m, piv+i);
//					int info;
//					My_dgetrf2(m-i,ib,A+(i+i*m), m, piv+i, &info);
					assert(info==0);

					for (int k=i; k<min(m,i+ib); k++)
						piv[k] += i;
				}

				#pragma omp task depend(out: dep[i/ib][0:i/ib]) depend(in: piv[i:ib])
				{
//					#pragma omp critical
//					cout << "dlaswp1: out: dep[" << i/ib << "][0:" << i/ib << "], in: piv[" << i << ":" << ib << "]\n";

					// Apply interchanges to columns 0:i
					int info = LAPACKE_dlaswp(MKL_COL_MAJOR, i, A, m, i+1, i+ib, piv, 1);
					assert(info==0);
//					My_dlaswp( i, A, m, i+1, i+ib, piv, 1);
				}

				if (i+ib < n)
				{
					for (int j=i+ib; j<n; j+=ib)
					{
						int jb = min(n-j,nb);

						#pragma omp task depend(inout: dep[i/ib][j/jb]) depend(in: piv[i:ib])
						{
//							#pragma omp critical
//							cout << "dlaswp2: inout: dep[" << i/ib << "][" << j/jb << "], in: piv[" << i << ":" << ib << "]\n";

							// Apply interchanges to columns i+ib:n-1
							int info = LAPACKE_dlaswp(MKL_COL_MAJOR, jb, A+(j*m), m, i+1, i+ib, piv, 1);
							assert(info==0);
//							My_dlaswp( jb, A+(j*m), m, i+1, i+ib, piv, 1);
						}

						#pragma omp task depend(in: dep[i/ib][i/ib]) depend(inout: dep[i/ib][j/jb])
						{
//							#pragma omp critical
//							cout << "dtrsm: in: dep[" << i/ib << "][" << i/ib << "], inout: dep[" << i/ib << "][" << j/jb << "]\n";

							// Compute block row of U
							cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
									ib, jb, 1.0, A+(i+i*m), m, A+(i+j*m), m);
						}

						// Update trailing submatrix
						if (i+ib < m) {
							#pragma omp task depend(in: dep[i/ib+1:p-i/ib-1][i/ib], dep[i/ib][j/jb]) depend(inout: dep[i/ib+1:p-i/ib-1][j/jb])
							{
//								#pragma omp critical
//								cout << "dgemm: in: dep[" << i/ib+1 << ":" << p-i/ib-1 << "][" << i/ib << "] dep[" << i/ib << "][" << j/jb << "], inout: dep[" << i/ib+1 << ":" << p-i/ib-1 << "][" << j/jb << "]\n";

								cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
										m-i-ib, jb, ib, -1.0, A+(i+ib+i*m), m, A+(i+j*m), m, 1.0, A+(i+ib+j*m), m);
							}
						} // End of if
					} // End of J-loop
				} // End of if
			} // End of I-loop
		} // End of single region
	} // End of parallel region

	timer = omp_get_wtime() - timer;   // Timer stop

	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;

	////////// Debug mode //////////
	#ifdef DEBUG
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			U[i+j*m] = (j<i) ? 0.0 : A[i+j*m];

	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
			m, n, 1.0, A, m, U, m);

	// Apply interchanges to matrix A
	int info = LAPACKE_dlaswp(MKL_COL_MAJOR, n, OA, m, 1, n, piv, 1);
	assert(info==0);

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

	for (int i=0; i<p; i++)
		delete [] dep[i];
	delete [] dep;

	return EXIT_SUCCESS;
}
