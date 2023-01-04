#pragma once

#include <vector>
#include "../model/Datapoint.h"

std::vector<float> compute_heatmap(std::vector<Datapoint> points);
std::vector<float> get_gaussian_kernel(float sigma, int kernel_size);