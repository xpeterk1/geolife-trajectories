#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "../model/Trajectory.h"

std::vector<glm::vec2> compute_pois(std::vector<Trajectory> points);