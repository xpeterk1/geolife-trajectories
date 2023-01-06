#include "heatmap.h"

#define __CUDACC__
#define __CUDA_ARCH__ 860
#define BLOCKSIZE 256

#define PRECISION 3

#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <fstream>

Datapoint* input_buffer = 0;
float* kernel_buffer = 0;
float* output_buffer = 0;
float* max_buffer = 0;
int n;

#pragma region Reduction Funtions

__device__ static inline float sum_function(float& x, float& y)
{
	return x + y;
}

__device__ static inline float min_function(float& x, float& y)
{
	return min(x, y);
}

__device__ static inline float max_function(float& x, float& y)
{
	return max(x, y);
}

#pragma endregion

#pragma region Kernels

__global__ void kde_kernel(Datapoint* points, int n_point, float* kernel, int kernel_size, float* output_data, int precision, int mode_mask, int min_time, int max_time)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;
	int output_size = pow(10, precision);

	if (global_id > n_point) return;

	Datapoint point = points[global_id];

	// filtering applied, 2047 = all bits sets to 1 => all present, no filtering
	if (mode_mask != 2047)
	{
		if ((mode_mask & UNKNOWN) == 0 && point.mode == UNKNOWN) return;
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
	}

	// filter time
	if (point.time < min_time || point.time > max_time) return;

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

__global__ void reduction_kernel_warp(float* data, float* output_value, int N, reductionOperation operation)
{
	float(*reduction_function)(float&, float&);
	float accumulator;

	switch (operation)
	{
	case MIN:
		reduction_function = min_function;
		accumulator = FLT_MAX;
		break;
	case MAX:
		reduction_function = max_function;
		accumulator = 0;
		break;
	case SUM:
		reduction_function = sum_function;
		accumulator = 0;
		break;
	default:
		accumulator = 0;
	}

	if (threadIdx.x == 0)
	{
		for (int i = 0; i < N; i++)
		{
			accumulator = reduction_function(accumulator, data[i]);
		}

		output_value[0] = accumulator;
	}
}

__global__ void reduction_kernel_blocks(float* data, float* output, int N, int num_blocks, reductionOperation operation)
{
	float(*reduction_function)(float&, float&);

	switch (operation)
	{
	case MIN:
		reduction_function = min_function;
		break;
	case MAX:
		reduction_function = max_function;
		break;
	case SUM:
		reduction_function = sum_function;
		break;
	}

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int thread_id = threadIdx.x;

	//copy data to local shared memory
	__shared__ float dataChunk[BLOCKSIZE];

	if (id >= 0 && id < N)
	{
		dataChunk[thread_id] = data[id];
	}
	__syncthreads();

	//compute reduction of the block
	for (unsigned int stride = (blockDim.x / 2); stride > 32; stride /= 2) {
		__syncthreads();

		if (thread_id < stride && id + stride < N)
		{
			dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + stride]);
		}
	}

	// reduction within one warp = last warp in the block
	if (thread_id < 32) {
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 32]);
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 16]);
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 8]);
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 4]);
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 2]);
		dataChunk[thread_id] = reduction_function(dataChunk[thread_id], dataChunk[thread_id + 1]);
	}

	// last thread of each block writes to the output
	if (thread_id == 0) {
		output[blockIdx.x] = dataChunk[0];
	}
}

__global__ void log_kernel(float* data, int N) 
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;

	if (global_id > N) return;

	if (data[global_id] != 0.0f)
		data[global_id] = log(data[global_id]);
}

__global__ void normalize_kernel(float* data, int N, float max)
{
	int global_id = threadIdx.x + blockDim.x * blockIdx.x;

	if (global_id > N) return;

	data[global_id] = data[global_id] / max;
}

#pragma endregion

void init_heatmap_data(std::vector<Datapoint> points, HeatmapConfig& config) 
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaError_t cudaStatus = cudaSetDevice(0);

	n = points.size();
	int output_size = pow(pow(10, PRECISION), 2);

	int size_bytes = n * sizeof(Datapoint);

	cudaMalloc((void**)&input_buffer, size_bytes);
	cudaMemcpy(input_buffer, &points[0], size_bytes, cudaMemcpyHostToDevice);

	// malloc output data float array of size (10^p)x(10^p)
	int output_size_bytes = output_size * sizeof(float);
	cudaMalloc((void**)&output_buffer, output_size_bytes);

	int grid_size = ceil((float)n / BLOCKSIZE);
	cudaMalloc((void**)&max_buffer, grid_size * sizeof(float));

	reinit_kernel(config);
}

void free_heatmap_data() 
{
	// free up used resources
	cudaFree(input_buffer);
	cudaFree(kernel_buffer);
	cudaFree(output_buffer);
	cudaFree(max_buffer);
}

std::vector<float> compute_heatmap(HeatmapConfig& config)
{
	int output_size = pow(pow(10, PRECISION), 2);
	int output_size_bytes = output_size * sizeof(float);

	cudaMemset(output_buffer, 0, output_size_bytes);

	int blocks = (floor(n) / BLOCKSIZE) + 1;

	// perform kernel density estimation
	kde_kernel << < blocks, BLOCKSIZE >> > (input_buffer, n, kernel_buffer, config.kernel_size, output_buffer, PRECISION, config.current_mode, config.min_time, config.max_time);
	
	if (config.use_log_scale && config.current_mode != 0)
		log_kernel << <blocks, BLOCKSIZE >> > (output_buffer, output_size);
	
	cudaDeviceSynchronize();

	// using parallel reduction, find maximal value in output_buffer
	// run the kernel -> local reduction results on each grid_size th element
	int grid_size = ceil((float)output_size / BLOCKSIZE);
	if (n >= 64)
	{
		// Array is long enough to be processed in blocks
		reduction_kernel_blocks << <grid_size, BLOCKSIZE >> > (output_buffer, max_buffer, output_size, grid_size, MAX);

		// input array is longer than one block
		if (grid_size > 1)
		{
			for (grid_size; grid_size >= 1; grid_size = ceil((float)grid_size / BLOCKSIZE))
			{
				reduction_kernel_blocks << <grid_size, BLOCKSIZE >> > (max_buffer, max_buffer, grid_size, ceil((float)grid_size / BLOCKSIZE), MAX);
				if (grid_size == 1)
					break;
			}
		}
	}
	else
	{
		// Array is too short for more complicated computation
		reduction_kernel_warp << <1, output_size >> > (output_buffer, max_buffer, output_size, MAX);
	}

	cudaDeviceSynchronize();

	float maxValue;
	cudaMemcpy(&maxValue, max_buffer, sizeof(float), cudaMemcpyDeviceToHost);

	// use previously found maxima to map all values to the range 0-1
	grid_size = ceil((float)output_size / BLOCKSIZE);
	normalize_kernel << <grid_size, BLOCKSIZE >> > (output_buffer, output_size, maxValue);
	cudaDeviceSynchronize();

	// copy results to CPU side
	std::vector<float> results;
	results.resize(output_size);
	cudaMemcpy(&results[0], output_buffer, output_size_bytes, cudaMemcpyDeviceToHost);

	return results;
}

void reinit_kernel(HeatmapConfig& config)
{
	cudaFree(kernel_buffer);
	int kernel_size_bytes = pow(config.kernel_size, 2) * sizeof(float);
	auto gauss_kernel = get_gaussian_kernel(config.sigma, config.kernel_size);

	cudaMalloc((void**)&kernel_buffer, kernel_size_bytes);
	cudaMemcpy(kernel_buffer, &gauss_kernel[0], kernel_size_bytes, cudaMemcpyHostToDevice);
}