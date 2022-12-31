#include "Dataset.h"

Dataset::Dataset(std::string path) 
{
	this->path = path;
	this->data = TrajectoryLoader().Load(path);
	this->size = data.size();
}


Dataset::Dataset(std::string path, int max_count) 
{
	this->path = path;
	this->data = TrajectoryLoader().Load(path, max_count);
	this->size = data.size();
}

