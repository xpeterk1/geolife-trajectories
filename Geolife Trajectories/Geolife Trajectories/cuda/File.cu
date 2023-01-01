
#define __CUDACC__
#define __CUDA_ARCH__ 860
#define NUMTHREADS 256

#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include "File.h"

__global__ void kernel(int* input) 
{
	input[0] = 123;
	__syncthreads();
}

void ahoj()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaError_t cudaStatus = cudaSetDevice(0);

	int pole[100];

	int* a;
	cudaMalloc((void**)&a, sizeof(int));
	kernel <<< 1, 1 >>> (a);
	
	int outputval;
	cudaStatus = cudaMemcpy(&outputval, a, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(a);
}