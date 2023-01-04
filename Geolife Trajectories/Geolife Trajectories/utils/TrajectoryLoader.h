#pragma once

#include "../model/Dataset.h"
#include "../model/Datapoint.h"
#include "../model/Labels.h"

#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>
#include <unordered_map>



class TrajectoryLoader
{

private:
	std::string data_path;
	Labels labels;

	// SMALLER BLOCK
	// constants used to filter out outliers
	const float LATITUDE_MAX = 40.1f;
	const float LATITUDE_MIN = 39.75f;
	const float LONGITUDE_MAX = 116.6f;
	const float LONGITUDE_MIN = 116.15f;

	//// LARGE BLOCK
	//const float LATITUDE_MAX = 40.8f;
	//const float LATITUDE_MIN = 39.4f;
	//const float LONGITUDE_MAX = 117.4f;
	//const float LONGITUDE_MIN = 115.8f;

	std::unordered_map<std::string, TransportationMode> const transportation_modes_table =
	{
		{"walk", TransportationMode::WALK},
		{"bike", TransportationMode::BIKE},
		{"bus", TransportationMode::BUS},
		{"car", TransportationMode::CAR},
		{"taxi", TransportationMode::CAR},
		{"subway", TransportationMode::SUBWAY},
		{"train", TransportationMode::TRAIN},
		{"airplane", TransportationMode::AIRPLANE},
		{"boat", TransportationMode::BOAT},
		{"run", TransportationMode::RUN},
		{"motorcycle", TransportationMode::MOTORCYCLE}
	};

public:

private:
	std::vector<Datapoint> ProcessTrajectory(std::filesystem::path path, bool normalize, Labels labels);
	std::vector<Datapoint> LoadFromTxt(std::string data_folder_path, int count, bool normalize);
	std::vector<Datapoint> LoadFromFolder(std::string data_folder_path, int count, bool normalize);
	std::vector<Datapoint> LoadFromBinary(std::string data_folder_path, int count);
	Labels ReadLabels(std::string labels_location);

public:
	std::vector<Datapoint> Load(std::string data_folder_path, bool normalize);
	std::vector<Datapoint> Load(std::string data_folder_path, int count, bool normalize);

};

