#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include <conio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define THREADSX 32
#define THREADSY 32


#define cudaCheckError() {										\
	cudaError_t e = cudaGetLastError();							\
	if (e != cudaSuccess) {										\
		printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,	\
		cudaGetErrorString(e));									\
		exit(1);												\
	}															\
}


__global__ void MatrixMulKernel(double *A_d, double *C_d, int m, int n) {

	int row = blockIdx.y * blockDim.y +  threadIdx.y;
	int col = blockIdx.x * blockDim.x +  threadIdx.x;
	int k;

	if ((row < n) && (col < n))
	{
		double sum = 0;
		for (k = 0; k < m; k++)
		{
			sum += A_d[k*n + col] * A_d[k*n + row ];
		}
		C_d[row*n + col] = sum;
	}
}


void MatrixMultiplication(double *a, double *mul, int m, int n) {

	double BlockX;
	double BlockY;


//	int size = m * n * sizeof(double);
//	printf("Size of A:%d\n",size);
//	long int size2 = n * n * sizeof(double);

//	printf("Size of Product:%ld\n",size2);
	double *A_d, *C_d;


	cudaMalloc((void**)&A_d, m * n * sizeof(double));
	cudaCheckError();

	cudaMemcpy(A_d, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMalloc((void**)&C_d, n * n * sizeof(double));
	cudaCheckError();
	//dim3 grid(1, 1);

	//threadX = ceil(double(n / BLOCKX));
	//threadY = ceil(double(n / BLOCKY));

	BlockX = ceil((double)n / THREADSX);
	BlockY = ceil((double)n / THREADSY);


	if (THREADSX >= n)
	{
		BlockX = 1;
	}
	if (THREADSY >= n)
	{
		BlockY = 1;
	}

	dim3 grid(BlockX,BlockY);
	dim3 block(THREADSX, THREADSY);

	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	MatrixMulKernel <<< grid, block >>>(A_d, C_d, m, n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time to generate :%f ms \n", time);

	cudaCheckError();

	cudaMemcpy(mul, C_d, n * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaFree(A_d);
	cudaCheckError();
	cudaFree(C_d);
	cudaCheckError();

}

void CpuMatrixMultiplication(double *A, double *C, int m, int n){
	double *mul_2;

	mul_2 = (double*)malloc(sizeof(double)*n*n);
	int i, j;
	double diff = 0;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			mul_2[i*n + j] = 0;
			for (int k = 0; k < m; k++)
			{
				mul_2[i*n + j] = mul_2[i*n + j] + A[n*k + i] * A[k*n + j];

			}
			// printf("A[%d][%d]--->%lf", i, j, A[i*n + j]);
			// printf("C[%d][%d]--->%lf\t", i, j,  C[i*n + j]);
		}
		// printf("\n");
	}

	/* CALCULATE THE DIFFERENCE BETWEEN THE MATRIX CALCULATED FROM GPU AND THE MATRIX CALCULATED IN CPU*/
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (diff != 0)
			{
				break;
			}
			else
			{
				diff += C[i*n + j] - mul_2[i*n + j];
			}
		}
		if (diff != 0)
		{
			break;
		}
	}

	printf("diff --> %lf\n", diff);
	free(mul_2);
}


int main(int argc,char* argv[]) {


	int m, n, i, j;

	//printf("Enter order of matrix A: ");
	//scanf_s("%d%d", &m, &n);

	if (argc == 3){
		m = atoi(argv[1]);
		n = atoi(argv[2]);
	}
	else{
			printf("Error: Invalid number of arguments");
			return 1;
	}

	double *A;
	double *C;
	time_t t;

	A = (double*)malloc(sizeof(double)*m*n);  // host memory for A
	C = (double*)malloc(sizeof(double)*n*n);  // host memory for C

	srand((unsigned)time(&t));

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			// printf("Enter value of a[%d][%d]: ", i, j);
			// scanf("%lf", &A[n*i + j]);
			A[n*i + j] = ((double)rand() / (double)RAND_MAX);
			
			// printf("A[%d][%d]-->%.4lf \t", i, j, A[n*i + j]);
		}
		// printf("\n");
	}

	printf("\tStart1\n" );
	MatrixMultiplication(A,C, m, n);

	// CpuMatrixMultiplication(A, C, m, n);


	free(C);
	free(A);

	//_getch();
	return 0;


}




