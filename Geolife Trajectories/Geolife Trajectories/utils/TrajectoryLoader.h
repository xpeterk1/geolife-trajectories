 #pragma once

#include "../model/Dataset.h"
#include "../model/Datapoint.h"

#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>



class TrajectoryLoader
{

private:
	std::string data_path;

	// constants used to filter out outliers
	const float LATITUDE_MAX = 40.1f;
	const float LATITUDE_MIN = 39.75f;
	const float LONGITUDE_MAX = 116.6f;
	const float LONGITUDE_MIN = 116.15f;

public:

private:
	std::vector<Datapoint> ProcessTrajectory(std::filesystem::path path, bool normalize);
	std::vector<Datapoint> LoadFromTxt(std::string data_folder_path, int count, bool normalize);
	std::vector<Datapoint> LoadFromFolder(std::string data_folder_path, int count, bool normalize);

public:
	std::vector<Datapoint> Load(std::string data_folder_path, bool normalize);
	std::vector<Datapoint> Load(std::string data_folder_path, int count, bool normalize);

};

