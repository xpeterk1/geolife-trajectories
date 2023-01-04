#include "Trajectory.h"

Trajectory::Trajectory(std::vector<glm::vec2> points, TransportationMode mode)
{
	positions = points;
	this->mode = mode;
	size = positions.size();
}