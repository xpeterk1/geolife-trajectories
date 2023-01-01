#include "Predict.h"

#define __CUDACC__
#define __CUDA_ARCH__ 860
#define NUMTHREADS 256

#define PRECISION 4

#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"

__global__ void kernel(glm::vec2* points, int n_point, float* kernel, int kernel_size)
{
	points[0].x = kernel[2];
	__syncthreads();
}

std::vector<glm::vec2> compute_pois(std::vector<Trajectory> points)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaError_t cudaStatus = cudaSetDevice(0);

	glm::vec2* input_buffer = 0;
	float* kernel_buffer = 0;
	int n = points[0].positions.size();
	int size_bytes = n * sizeof(glm::vec2);
	int kernel_size = 5;
	int kernel_size_bytes = pow(kernel_size, 2) * sizeof(float);

	auto gauss_kernel = get_gaussian_kernel( 1, kernel_size);

	cudaMalloc((void**)&input_buffer, size_bytes);
	cudaMemcpy(input_buffer, & points[0].positions[0], size_bytes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&kernel_buffer, kernel_size_bytes);
	cudaMemcpy(kernel_buffer, &gauss_kernel[0], kernel_size_bytes, cudaMemcpyHostToDevice);

	kernel << < 1, 1 >> > (input_buffer, n, kernel_buffer, kernel_size);

	std::vector<glm::vec2> results;
	results.resize(n);
	cudaMemcpy(&results[0], input_buffer, size_bytes, cudaMemcpyDeviceToHost);
	
	// free up used resources
	cudaFree(input_buffer);
	cudaFree(kernel_buffer);

	return std::vector<glm::vec2>();
}