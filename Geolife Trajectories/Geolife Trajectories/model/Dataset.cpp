#include "Dataset.h"

Dataset::Dataset(std::string path, bool normalize) 
{
	this->path = path;
	this->data = TrajectoryLoader().Load(path, normalize);
	this->size = data.size();
}


Dataset::Dataset(std::string path, int max_count, bool normalize)
{
	this->path = path;
	this->data = TrajectoryLoader().Load(path, max_count, normalize);
	this->size = data.size();
}

