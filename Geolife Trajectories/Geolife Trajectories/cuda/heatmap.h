#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "../model/Datapoint.h"

std::vector<glm::vec2> compute_heatmap(std::vector<Datapoint> points);
std::vector<float> get_gaussian_kernel(float sigma, int kernel_size);