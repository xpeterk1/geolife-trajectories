#include "Predict.h"

#define __CUDACC__
#define __CUDA_ARCH__ 860
#define BLOCKSIZE 256

#define PRECISION 4
#define KERNEL_SIZE 9
#define SIGMA 1.0f

#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"

/// <summary>
/// Perform Kernel Density Estimation using Gaussian Kernel
/// WARNING - output data are NOT normalized to have an integral of 1 -> it's not a true density function
/// </summary>
/// <param name="points">All input points</param>
/// <param name="n_point">Total number of points</param>
/// <param name="kernel">Kernel in 1D array</param>
/// <param name="kernel_size">Size of kernel (one side)</param>
/// <param name="output_data">Output data 2D matrix of size (10^precision)^2</param>
/// <param name="precision">Precision of output data</param>
/// <returns></returns>
__global__ void kde_kernel(glm::vec2* points, int n_point, float* kernel, int kernel_size, float* output_data, int precision)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;
	int output_size = pow(10, precision);

	if (global_id > n_point) return;

	// find corresponding (x, y) coordinates in the output_data
	int x = (int)(points[global_id].x * output_size) + 1;
	int y = (int)(points[global_id].y * output_size) + 1;
	
	// perform addition around (x, y)
	int kernel_index = 0;
	for (int u = -kernel_size / 2; u <= kernel_size / 2; u++) 
	{
		for (int v = -kernel_size / 2; v <= kernel_size / 2; v++) 
		{
			float kernel_value = kernel[kernel_index];
			int output_index = (y + u) * output_size + (x + v);

			// output index out of bounds of output data
			if (output_index < 0 || output_index >= pow(output_size, 2)) continue;

			atomicAdd(&output_data[output_index], kernel_value);
			kernel_index++;
		}
	}

	__threadfence();
}

std::vector<glm::vec2> compute_pois(std::vector<Trajectory> points)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaError_t cudaStatus = cudaSetDevice(0);

	glm::vec2* input_buffer = 0;
	float* kernel_buffer = 0;
	float* output_buffer = 0;

	int n = points[0].positions.size();
	int output_size = pow(pow(10, PRECISION), 2);

	int size_bytes = n * sizeof(glm::vec2);
	int kernel_size_bytes = pow(KERNEL_SIZE, 2) * sizeof(float);

	auto gauss_kernel = get_gaussian_kernel( SIGMA, KERNEL_SIZE);

	cudaMalloc((void**)&input_buffer, size_bytes);
	cudaMemcpy(input_buffer, & points[0].positions[0], size_bytes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&kernel_buffer, kernel_size_bytes);
	cudaMemcpy(kernel_buffer, &gauss_kernel[0], kernel_size_bytes, cudaMemcpyHostToDevice);

	// malloc output data float array of size (10^p)x(10^p)
	int output_size_bytes = output_size * sizeof(float);
	cudaMalloc((void**)&output_buffer, output_size_bytes);

	//TODO: spravny pocet vlaken
	int blocks = (floor(n) / BLOCKSIZE) + 1;
	kde_kernel << < blocks, BLOCKSIZE >> > (input_buffer, n, kernel_buffer, KERNEL_SIZE, output_buffer, PRECISION);

	// copy results to CPU side
	std::vector<float> results;
	results.resize(output_size);
	cudaMemcpy(&results[0], output_buffer, output_size_bytes, cudaMemcpyDeviceToHost);

	// free up used resources
	cudaFree(input_buffer);
	cudaFree(kernel_buffer);
	cudaFree(output_buffer);

	return std::vector<glm::vec2>();
}