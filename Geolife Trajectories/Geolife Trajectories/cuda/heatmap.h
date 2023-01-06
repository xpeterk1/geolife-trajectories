#pragma once

#include <vector>
#include "../model/Datapoint.h"
#include "../model/HeatmapConfig.h"

enum reductionOperation {
	MIN, MAX, SUM
};

std::vector<float> compute_heatmap(HeatmapConfig& config);
std::vector<float> get_gaussian_kernel(float sigma, int kernel_size);
void free_heatmap_data();
void init_heatmap_data(std::vector<Datapoint> points, HeatmapConfig& config);
void reinit_kernel(HeatmapConfig& config);