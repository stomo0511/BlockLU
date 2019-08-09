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

// Set all the elements of vector as Val
void Set_vec_elements(const int m, double *b, const double Val)
{
//	#pragma omp parallel for
	for (int i=0; i<m; i++)
		b[i] = Val;
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

	double *A = new double[m*n];     // Original matrix
	double *b = new double[m];       // Right-hand vector
	int *piv = new int[m];           // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix
	Set_vec_elements(m,b,1.0);       // Set all the elements of vec. b as 1.0

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	Copy_mat(m,n,A,OA);
	double *U = new double[m*n];
	#endif

	cout << "A : \n";
	Show_mat(m,n,A);

	double timer = omp_get_wtime();

	for (int i=0; i<n; i+=nb)
	{
		int ib = min(n-i,nb);

		int info = LAPACKE_dgetrf2(MKL_COL_MAJOR, m-i, ib, A+(i+i*m), m, piv+i);
		assert(info==0);

		for (int k=i; k<min(m,i+ib); k++)
			piv[k] += i;

		// Apply interchanges to columns 0:i
		info = LAPACKE_dlaswp(MKL_COL_MAJOR, i, A, m, i+1, i+ib, piv, 1);
		assert(info==0);

		if (i+ib < n)
		{
			// Apply interchanges to columns i+ib:n-1
			info = LAPACKE_dlaswp(MKL_COL_MAJOR, n-i-ib, A+((i+ib)*m), m, i+1, i+ib, piv, 1);
			assert(info==0);

			// Compute block row of U
			cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
					ib, n-i-ib, 1.0, A+(i+i*m), m, A+(i+(i+ib)*m), m);

			// Update trailing submatrix
			if (i+ib < m) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
						m-i-ib, n-i-ib, ib, -1.0, A+(i+ib+i*m), m, A+(i+(i+ib)*m), m, 1.0, A+(i+ib+(i+ib)*m), m);
			}
		}
	}

	timer = omp_get_wtime() - timer;

	cout << "Result : \n";
	Show_mat(m,n,A);
//	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;

	cout << "piv : \n";
	for (int i=0; i<m; i++)
		cout << piv[i] << ", ";
	cout << endl;

	////////// Debug mode //////////
	#ifdef DEBUG
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			U[i+j*m] = (j<i) ? 0.0 : A[i+j*m];

//	cout << "U : \n";
//	Show_mat(m,n,U);

	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
			m, n, 1.0, A, m, U, m);

	cout << "L*U : \n";
	Show_mat(m,n,U);

	// Apply interchanges to columns 0:i
	int info = LAPACKE_dlaswp(MKL_COL_MAJOR, n, U, m, 1, n, piv, 1);
	assert(info==0);

	cout << "P*L*U : \n";
	Show_mat(m,n,U);

	double tmp = 0.0;
	for (int i=0; i<m*n; i++)
		tmp += (OA[i] - U[i])*(OA[i] - U[i]);

	cout << "|| A - LU ||_2 = " << sqrt(tmp) << endl;

	delete [] OA;
	delete [] U;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] b;
	delete [] piv;

	return EXIT_SUCCESS;
}
