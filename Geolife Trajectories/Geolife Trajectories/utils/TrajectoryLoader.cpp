#include "TrajectoryLoader.h"

std::vector<Trajectory> TrajectoryLoader::Load(std::string data_folder_path, bool normalize)
{
	data_path = data_folder_path;
	return Load(data_folder_path, -1, normalize);
}

std::vector<Trajectory> TrajectoryLoader::Load(std::string data_source_path, int count, bool normalize)
{
	data_path = data_source_path;

	std::filesystem::directory_entry entry = std::filesystem::directory_entry(data_source_path);

	if (entry.is_directory())
		return LoadFromFolder(data_source_path, count, normalize);
	else
		return LoadFromTxt(data_source_path, count, normalize);
}

Trajectory TrajectoryLoader::ProcessTrajectory(std::filesystem::path path, bool normalize)
{
	std::ifstream file(path);

	std::vector<glm::vec2> points;
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
		std::string token;

		pos = line.find(",");
		token = line.substr(0, pos);
		float latitude = std::stof(token);
		line.erase(0, pos + 1);

		pos = line.find(",");
		token = line.substr(0, pos);
		float longitude = std::stof(token);
		line.erase(0, pos + 1);

		// point falls outside of the examined map piece
		if (latitude < LATITUDE_MIN || longitude < LONGITUDE_MIN || latitude > LATITUDE_MAX || longitude > LONGITUDE_MAX) continue;

		if (normalize)
		{
			latitude = (latitude - LATITUDE_MIN) / (LATITUDE_MAX - LATITUDE_MIN);
			longitude = (longitude - LONGITUDE_MIN) / (LONGITUDE_MAX - LONGITUDE_MIN);
		}

		points.push_back(glm::vec2(latitude, longitude));
	}

	file.close();
	return Trajectory(points, UNKNOWN);
}

std::vector<Trajectory> TrajectoryLoader::LoadFromTxt(std::string data_txt_path, int count, bool normalize) 
{
	std::ifstream file(data_txt_path);
	std::vector<Trajectory> output;
	std::vector<glm::vec2> points;
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

		points.push_back(glm::vec2(latitude, longitude));
		counter++;
	}

	output.push_back(Trajectory(points, UNKNOWN));
	return output;
}

std::vector<Trajectory> TrajectoryLoader::LoadFromFolder(std::string data_folder_path, int count, bool normalize)
{
	std::vector<Trajectory> output;
	int counter = 0;
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(data_folder_path))
	{
		// enough entires collected
		if (counter == count) break;

		// current entry refers to a directory
		if (dirEntry.is_directory()) continue;

		if (dirEntry.path().extension() == ".txt" || dirEntry.path().extension() == ".pdf") continue;

		Trajectory traj = ProcessTrajectory(dirEntry.path(), normalize);
		if (traj.size > 0)
			output.push_back(traj);

		counter++;
	}

	return output;
}