#include "heatmap.h"

#define __CUDACC__
#define __CUDA_ARCH__ 860
#define BLOCKSIZE 256

#define PRECISION 3
#define KERNEL_SIZE 3
#define SIGMA 1.0f

#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <fstream>

Datapoint* input_buffer = 0;
float* kernel_buffer = 0;
float* output_buffer = 0;
int n;

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
/// /// <param name="mode_mask">Bit mask for transportation mode</param>
/// <returns></returns>
__global__ void kde_kernel(Datapoint* points, int n_point, float* kernel, int kernel_size, float* output_data, int precision, int mode_mask)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;
	int output_size = pow(10, precision);

	if (global_id > n_point) return;

	Datapoint point = points[global_id];

	// filtering applied, 2047 = all bits sets to 1 => all present, no filtering
	if (mode_mask != 2047)
	{
		if ((mode_mask & WALK) == 0 && point.mode == WALK) return;
		if ((mode_mask & BIKE) == 0 && point.mode == BIKE) return;
		if ((mode_mask & BUS) == 0 && point.mode == BUS) return;
		if ((mode_mask & CAR) == 0 && point.mode == CAR) return;
		if ((mode_mask & SUBWAY) == 0 && point.mode == SUBWAY) return;
		if ((mode_mask & TRAIN) == 0 && point.mode == TRAIN) return;
		if ((mode_mask & AIRPLANE) == 0 && point.mode == AIRPLANE) return;
		if ((mode_mask & BOAT) == 0 && point.mode == BOAT) return;
		if ((mode_mask & RUN) == 0 && point.mode == RUN) return;
		if ((mode_mask & MOTORCYCLE) == 0 && point.mode == MOTORCYCLE) return;
		if (point.mode == UNKNOWN) return;
	}

	// find corresponding (x, y) coordinates in the output_data
	// AND SWITCH X AND Y
	int x = (int)(point.y * output_size) + 1;
	int y = (int)(point.x * output_size) + 1;
	
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

__global__ void normalize_kernel(float* data, int data_size, int N)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (global_id > data_size) return;

	data[global_id] /= float(N);
}

void init_heatmap_data(std::vector<Datapoint> points) 
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaError_t cudaStatus = cudaSetDevice(0);

	n = points.size();
	int output_size = pow(pow(10, PRECISION), 2);

	int size_bytes = n * sizeof(Datapoint);
	int kernel_size_bytes = pow(KERNEL_SIZE, 2) * sizeof(float);

	auto gauss_kernel = get_gaussian_kernel(SIGMA, KERNEL_SIZE);

	cudaMalloc((void**)&input_buffer, size_bytes);
	cudaMemcpy(input_buffer, &points[0], size_bytes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&kernel_buffer, kernel_size_bytes);
	cudaMemcpy(kernel_buffer, &gauss_kernel[0], kernel_size_bytes, cudaMemcpyHostToDevice);

	// malloc output data float array of size (10^p)x(10^p)
	int output_size_bytes = output_size * sizeof(float);
	cudaMalloc((void**)&output_buffer, output_size_bytes);
}

void free_heatmap_data() 
{
	// free up used resources
	cudaFree(input_buffer);
	cudaFree(kernel_buffer);
	cudaFree(output_buffer);
}

std::vector<float> compute_heatmap(int mode_mask)
{
	int output_size = pow(pow(10, PRECISION), 2);
	int output_size_bytes = output_size * sizeof(float);

	cudaMemset(output_buffer, 0, output_size_bytes);

	int blocks = (floor(n) / BLOCKSIZE) + 1;
	kde_kernel << < blocks, BLOCKSIZE >> > (input_buffer, n, kernel_buffer, KERNEL_SIZE, output_buffer, PRECISION, mode_mask);
	//normalize_kernel << <blocks, BLOCKSIZE >> > (output_buffer, n, points.size());

	// copy results to CPU side
	std::vector<float> results;
	results.resize(output_size);
	cudaMemcpy(&results[0], output_buffer, output_size_bytes, cudaMemcpyDeviceToHost);

	return results;
}