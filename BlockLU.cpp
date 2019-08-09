//============================================================================
// Name        : 04_BlockLU.cpp
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

	double *A = new double[m*n];   // Original matrix
	int *piv = new int[m];          // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	Copy_mat(m,n,A,OA);
	double *U = new double[m*n];
	#endif

	double timer = omp_get_wtime();

	LAPACKE_dgetrf( MKL_COL_MAJOR, m, n, A, m, piv );

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

	return EXIT_SUCCESS;
}
