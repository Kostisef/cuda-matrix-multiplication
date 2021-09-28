#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "curand.h"

#include <stdio.h>
//#include <conio.h>
#include <stdlib.h>
#include <time.h>

#define RANDOM_MAX 10000

#define cudaCheckError() {											\
	cudaError_t e = cudaGetLastError();								\
	if (e != cudaSuccess) {											\
		printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,		\
							cudaGetErrorString(e));					\
		exit(1);													\
	}																\
}


const char* cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

void cublasCheckError(cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS error %s:%d: %s\n", __FILE__, __LINE__, cublasGetErrorString(status));
			exit(1);
	}
}


void CpuMatrixMultiplication(double *A,double *C ,int m, int n){
	double *mul_2;

	mul_2 = (double*)malloc(sizeof(double)*n*n);
	int i, j;
	int diff = 0;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			mul_2[i*n + j] = 0;
			for (int k = 0; k < m; k++)
			{
				mul_2[i*n + j] = mul_2[i*n + j] + A[n*k + i] * A[k*n + j];

			}
			//printf("A[%d][%d]--->%lf", i, j, A[i*n + j]);
			//printf("C[%d][%d]--->%lf\t", i, j, mul_2[i*n + j]);
		}
		//printf("\n");
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

	printf("diff --> %lf \n", diff);
	free(mul_2);
}


int CublasMatrixMultiplication(double *A, double *C, int m, int n){
	
	cublasStatus_t stat;				// CUBLAS functions status
	cublasHandle_t handle;				// CUBLAS context

	double *d_A;
	double *d_C;

	cudaMalloc((void**)&d_A, sizeof(double)* m * n); // device memory alloc for A
	cudaCheckError();
	cudaMalloc((void**)&d_C, sizeof(double)* n * n); // device memory alloc for C
	cudaCheckError();

	stat = cublasCreate(&handle);  // initialize CUBLAS context
	cublasCheckError(stat);

	stat = cublasSetMatrix(m, n, sizeof (double), A, m, d_A, m);  // A -> d_A
	cublasCheckError(stat);

	stat = cublasSetMatrix(n, n, sizeof (double), C, n, d_C, n);  // C -> d_C
	cublasCheckError(stat);

	double alpha = 1.0;
	double beta = 0.0;


	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &alpha, d_A, n, d_A, n, &beta, d_C, n);
	cublasCheckError(stat);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time : %f ms \n", time);

	stat = cublasGetMatrix(n, n, sizeof (double), d_C, n, C, n);
	cublasCheckError(stat);

	cudaFree(d_A); // free device memory
	cudaFree(d_C); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	printf("\n");
	printf("\n");
	return EXIT_SUCCESS;

}

int main(int argc,char* argv[]) {

	int ret;
	int m, n, j,i;

// 	int nDevices;

//   cudaGetDeviceCount(&nDevices);
//   for (i = 0; i < nDevices; i++) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, i);
//     printf("Device Number: %d\n", i);
//     printf("  Device name: %s\n", prop.name);
//     printf("  Memory Clock Rate (KHz): %d\n",
//            prop.memoryClockRate);
//     printf("  Memory Bus Width (bits): %d\n",
//            prop.memoryBusWidth);
//     printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
//            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
// }

	//int diff = 0;

	//printf("Enter order of matrix A: ");
	//scanf("%d%d", &m, &n);

	if (argc == 2){
		m = 5000;
		n = atoi(argv[1]);
	}
	else{
		printf("Error: Invalid number of arguments\n");
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
			//printf("Enter value of a[%d][%d]: ", i, j);
			//scanf("%lf", &A[n*i + j]);
			A[n*i + j] = ((double)rand() / (double)RANDOM_MAX);
			//A[n*i+j] = (double)rand();
			//printf("A[%d][%d]-->%.4lf \t", i, j, A[n*i + j]);
		}
		//printf("\n");
	}

	printf("\tStart\n" );
	ret = CublasMatrixMultiplication(A, C, m, n);

	//CpuMatrixMultiplication(A, C, m, n);


	free(A); // free host memory
	free(C); // free host memory
	
	//_getch();
	return ret;


}


