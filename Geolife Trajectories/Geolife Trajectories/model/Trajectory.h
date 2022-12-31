#pragma once

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

enum TransportationMode
{
	UNKNOWN, WALK, BIKE, BUS, CAR, TRAIN, AIRPLANE, OTHER
};

class Trajectory
{

private:
	std::vector<glm::vec2> positions;

public:
	int size;
	TransportationMode mode;

private:

public:
	Trajectory(std::vector<glm::vec2> positions, TransportationMode mode);

};

