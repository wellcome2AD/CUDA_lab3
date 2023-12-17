#include <cassert>
#include <iostream>
#include <omp.h>

void printMatrix(float* matrix, size_t n, size_t m)
{
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < m; ++j)
		{
			std::cout << matrix[i * m + j] << ' ';
		}
		std::cout << std::endl;
	}
}

void multiple(float* A, size_t n1, size_t m1, float* B, size_t n2, size_t m2, float* C)
{
	assert(m1 == n2);
	auto&& c_m = m2;
	for (size_t i = 0; i < n1; ++i)
	{
		for (size_t j = 0; j < m2; ++j)
		{
			C[i * c_m + j] = 0;
			for (int l = 0; l < m1; l++) {
				C[i * c_m + j] += (A[i * m1 + l] * B[l * m2 + j]);
			}
		}
	}
}

void multipleOMP(float* A, size_t n1, size_t m1, float* B, size_t n2, size_t m2, float* C)
{
	assert(m1 == n2);
	auto&& c_m = m2;
#pragma omp parallel for
	for (size_t i = 0; i < n1; ++i)
	{
#pragma omp parallel for
		for (size_t j = 0; j < m2; ++j)
		{
			C[i * c_m + j] = 0;
#pragma omp parallel for
			for (int l = 0; l < m1; l++) {
				C[i * c_m + j] += (A[i * m1 + l] * B[l * m2 + j]);
			}
		}
	}
}

int main()
{
	const size_t m = 1000, n = 1000, k = 1000;
	float* A = new float[m * n], *B = new float[n * k], *C = new float[m * k], *Cexp = new float[m * k];

	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			A[i * n + j] = j + 1;
		}
	}
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			B[i * k + j] = j + 1;
		}
	}
		
	multiple(A, m, n, B, n, k, Cexp);
	auto start = omp_get_wtime();
	multipleOMP(A, m, n, B, n, k, C);
	auto end = omp_get_wtime() - start;
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			assert(C[i * k + j] == Cexp[i * k + j]);
		}
	}
	std::cout.precision(8);
	std::cout << "Work time: " << end << std::endl;
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] Cexp;
	return 0;
}
