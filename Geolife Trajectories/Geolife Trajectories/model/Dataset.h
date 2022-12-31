#pragma once

#include "Trajectory.h"
#include "../utils/TrajectoryLoader.h"

#include <vector>
#include <string>

class Dataset
{

private:

public:
	std::string path;
	std::vector<Trajectory> data;
	int size;

private:

public:
	Dataset(std::string path, bool normalize);
	Dataset(std::string path, int max_count, bool normalize);

};