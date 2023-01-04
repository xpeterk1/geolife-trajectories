#include "TrajectoryLoader.h"

std::vector<Datapoint> TrajectoryLoader::Load(std::string data_folder_path, bool normalize)
{
	data_path = data_folder_path;
	return Load(data_folder_path, -1, normalize);
}

std::vector<Datapoint> TrajectoryLoader::Load(std::string data_source_path, int count, bool normalize)
{
	data_path = data_source_path;

	std::filesystem::directory_entry entry = std::filesystem::directory_entry(data_source_path);

	if (entry.is_directory())
		return LoadFromFolder(data_source_path, count, normalize);
	else
		return LoadFromTxt(data_source_path, count, normalize);
}

std::vector<Datapoint> TrajectoryLoader::ProcessTrajectory(std::filesystem::path path, bool normalize)
{
	std::ifstream file(path);

	std::vector<Datapoint> points;
	std::string line;

	//skip first 6 lines
	for (int i = 0; i < 6; i++)
	{
		std::getline(file, line);
	}

	//read actual lines
	while (file)
	{
		std::getline(file, line);

		// EOF reached
		if (line.empty()) break;

		size_t pos = 0;
		std::string token[7];

		for (int i = 0; i < 7; i++) 
		{
			pos = line.find(",");
			token[i] = line.substr(0, pos);
			line.erase(0, pos + 1);
		}

		// longite + latitude
		float latitude = std::stof(token[0]);		
		float longitude = std::stof(token[1]);

		// time of day
		struct std::tm tm = {0};
		std::istringstream ss(token[6]);
		ss >> std::get_time(&tm, "%T"); // or just %T in this case
		int seconds = tm.tm_sec + tm.tm_min * 60 + tm.tm_hour * 3600;

		// point falls outside of the examined map piece
		if (latitude < LATITUDE_MIN || longitude < LONGITUDE_MIN || latitude > LATITUDE_MAX || longitude > LONGITUDE_MAX) continue;

		if (normalize)
		{
			latitude = (latitude - LATITUDE_MIN) / (LATITUDE_MAX - LATITUDE_MIN);
			longitude = (longitude - LONGITUDE_MIN) / (LONGITUDE_MAX - LONGITUDE_MIN);
		}

		points.push_back(Datapoint(latitude, longitude, UNKNOWN, time_t(seconds)));
	}

	file.close();
	return points;
}

std::vector<Datapoint> TrajectoryLoader::LoadFromTxt(std::string data_txt_path, int count, bool normalize)
{
	std::ifstream file(data_txt_path);
	std::vector<Datapoint> output;
	std::string line;
	int counter = 0;
	
	//read actual lines
	while (file)
	{
		// requested number of entries reached
		if (counter == count) break;

		std::getline(file, line);

		// EOF reached
		if (line.empty()) break;

		size_t pos = 0;
		std::string token;

		pos = line.find(", ");
		token = line.substr(0, pos);
		std::replace(token.begin(), token.end(), ',', '.');
		float latitude = std::stof(token);
		line.erase(0, pos + 2);

		pos = line.find(", ");
		token = line.substr(0, pos);
		std::replace(token.begin(), token.end(), ',', '.');
		float longitude = std::stof(token);
		line.erase(0, pos + 2);

		// point falls outside of the examined map piece
		if (latitude < LATITUDE_MIN || longitude < LONGITUDE_MIN || latitude > LATITUDE_MAX || longitude > LONGITUDE_MAX) continue;

		if (normalize)
		{
			latitude = (latitude - LATITUDE_MIN) / (LATITUDE_MAX - LATITUDE_MIN);
			longitude = (longitude - LONGITUDE_MIN) / (LONGITUDE_MAX - LONGITUDE_MIN);
		}

		output.push_back(Datapoint(latitude, longitude, UNKNOWN, time_t(-1)));
		counter++;
	}

	return output;
}

std::vector<Datapoint> TrajectoryLoader::LoadFromFolder(std::string data_folder_path, int count, bool normalize)
{
	std::vector<Datapoint> output;
	int counter = 0;
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(data_folder_path))
	{
		// enough entires collected
		if (counter >= count) break;

		// current entry refers to a directory
		if (dirEntry.is_directory()) continue;

		if (dirEntry.path().extension() == ".txt" || dirEntry.path().extension() == ".pdf") continue;

		std::vector<Datapoint> points = ProcessTrajectory(dirEntry.path(), normalize);
		if (points.size() > 0)
			output.insert(output.end(), points.begin(), points.end());

		counter += points.size();
	}

	return output;
}