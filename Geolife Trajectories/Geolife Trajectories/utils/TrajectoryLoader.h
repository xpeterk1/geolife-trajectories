 #pragma once

#include "../model/Dataset.h"

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>



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
	Trajectory ProcessTrajectory(std::filesystem::path path, bool normalize);
	std::vector<Trajectory> LoadFromTxt(std::string data_folder_path, int count, bool normalize);
	std::vector<Trajectory> LoadFromFolder(std::string data_folder_path, int count, bool normalize);

public:
	std::vector<Trajectory> Load(std::string data_folder_path, bool normalize);
	std::vector<Trajectory> Load(std::string data_folder_path, int count, bool normalize);

};

