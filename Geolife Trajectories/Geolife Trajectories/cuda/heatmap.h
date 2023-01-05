#pragma once

#include <vector>
#include "../model/Datapoint.h"

enum reductionOperation {
	MIN, MAX, SUM
};

std::vector<float> compute_heatmap(int mode_mask);
std::vector<float> get_gaussian_kernel(float sigma, int kernel_size);
void free_heatmap_data();
void init_heatmap_data(std::vector<Datapoint> points);