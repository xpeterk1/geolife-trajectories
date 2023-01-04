#pragma once

#include <vector>
#include "../model/Datapoint.h"

void compute_heatmap(std::vector<Datapoint> points);
std::vector<float> get_gaussian_kernel(float sigma, int kernel_size);