#include "Predict.h"

#define PI 3.14159265f

std::vector<float> get_gaussian_kernel(float sigma, int kernel_size)
{
	std::vector<float> kernel;
	kernel.resize(kernel_size * kernel_size);

	float sum = 0.0; // For accumulating the kernel values
	for (int y = -kernel_size / 2; y <= kernel_size / 2; y++)
	{
		for (int x = -kernel_size / 2; x <= kernel_size / 2; x++)
		{
			float val = 1 / (2 * PI * pow(sigma, 2)) * exp(-((pow(x, 2) + pow(y, 2))/(2 * pow(sigma, 2))));
			kernel[(y + kernel_size / 2) * kernel_size + (x + kernel_size / 2)] = val;
			sum += val;
		}
	}

	// Normalize the kernel
	//for (int i = 0; i < pow(kernel_size, 2); i++)
	//{
	//	kernel[i] /= sum;
	//}

	return kernel;
}