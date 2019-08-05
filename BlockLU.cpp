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

// Set all the elements of vector as Val
void Set_vec_elements(const int m, double *b, const double Val)
{
//	#pragma omp parallel for
	for (int i=0; i<m; i++)
		b[i] = Val;
}

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
	double *b = new double[m];     // Right-hand vector
	int *ipiv = new int[m];          // permutation vector

	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix
//	Show_mat(m,n,A);

	Set_vec_elements(m,b,1.0);       // Set all the elements of vec. b as 1.0

	double timer = omp_get_wtime();

	LAPACKE_dgetrf( MKL_COL_MAJOR, m, n, A, m, ipiv );

	timer = omp_get_wtime() - timer;

	Show_mat(m,n,A);
	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;

	delete [] A;
	delete [] b;
	delete [] ipiv;

	return EXIT_SUCCESS;
}
