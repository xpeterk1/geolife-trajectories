#pragma once

#include "Datapoint.h"
#include "../utils/TrajectoryLoader.h"

#include <vector>
#include <string>

class Dataset
{

private:

public:
	std::string path;
	std::vector<Datapoint> data;
	int size;

private:

public:
	Dataset(std::string path, bool normalize);
	Dataset(std::string path, int max_count, bool normalize);

};