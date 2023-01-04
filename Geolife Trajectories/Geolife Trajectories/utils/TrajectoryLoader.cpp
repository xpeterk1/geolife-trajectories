#include "TrajectoryLoader.h"

std::vector<Datapoint> TrajectoryLoader::Load(std::string data_folder_path, bool normalize)
{
	data_path = data_folder_path;
	return Load(data_folder_path, 9999999999, normalize);
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

std::vector<Datapoint> TrajectoryLoader::ProcessTrajectory(std::filesystem::path path, bool normalize, Labels labels)
{
	std::ifstream file(path);

	std::vector<Datapoint> points;
	std::string line;

	// skip first 6 lines
	for (int i = 0; i < 6; i++)
	{
		std::getline(file, line);
	}

	// read actual lines
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
		struct std::tm tm = { 0 };
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

		// Transportation mode
		TransportationMode mode = UNKNOWN;
		if (!labels.Empty()) 
		{
			// read date and time
			struct std::tm tm_start = { 0 };
			std::istringstream ss_start(token[5]);
			ss_start >> std::get_time(&tm_start, "%Y-%m-%d"); // or just %T in this case
			int date = tm_start.tm_year * 10000 + tm_start.tm_mon * 100 + tm_start.tm_mday;
		
			if (labels.labels.contains(date)) 
			{
				std::vector<Label> l = labels.labels.at(date);
				
				for (Label lab : l)
				{
					if (lab.start_seconds <= seconds && lab.end_seconds >= seconds)
					{
						mode = lab.mode;
						break;
					}
				}
			}
		}

		points.push_back(Datapoint(latitude, longitude, mode, time_t(seconds)));
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


	// look for labels.txt file in the current folder
	for (const auto& dirEntry : std::filesystem::directory_iterator(data_folder_path))
	{
		// file found - read it
		if (dirEntry.path().filename() == "labels.txt")
		{
			labels = ReadLabels(dirEntry.path().string());
		}
	}

	// iterate again and recursive read directory / file
	for (const auto& dirEntry : std::filesystem::directory_iterator(data_folder_path))
	{
		// current entry refers to a directory
		if (dirEntry.is_directory())
		{
			auto new_points = LoadFromFolder(dirEntry.path().string(), count - counter, normalize);
			output.insert(output.end(), new_points.begin(), new_points.end());
			counter += new_points.size();

			if (counter >= count) break;
		}

		// I don't want to read txt and pdf files again
		if (dirEntry.path().extension() == ".txt" || dirEntry.path().extension() == ".pdf") continue;

		// Current file is trajectory file
		std::vector<Datapoint> points = ProcessTrajectory(dirEntry.path(), normalize, labels);
		if (points.size() > 0)
			output.insert(output.end(), points.begin(), points.end());
	}

	labels.Clear();
	return output;
}

Labels TrajectoryLoader::ReadLabels(std::string labels_location)
{
	std::ifstream file(labels_location);

	std::string line;
	Labels labels;

	// Skip first line
	std::getline(file, line);

	while (file)
	{
		std::getline(file, line);

		// EOF reached
		if (line.empty()) break;

		size_t pos = 0;
		std::string token[3];

		for (int i = 0; i < 3; i++)
		{
			pos = line.find("\t");
			token[i] = line.substr(0, pos);
			line.erase(0, pos + 1);
		}

		struct std::tm tm_start = { 0 };
		std::istringstream ss_start(token[0]);
		ss_start >> std::get_time(&tm_start, "%Y/%m/%d %H:%M:%S"); // or just %T in this case
		int start_date = tm_start.tm_year * 10000 + tm_start.tm_mon * 100 + tm_start.tm_mday;
		int start_seconds = tm_start.tm_sec + tm_start.tm_min * 60 + tm_start.tm_hour * 3600;

		struct std::tm tm_end = { 0 };
		std::istringstream ss_end(token[1]);
		ss_end >> std::get_time(&tm_end, "%Y/%m/%d %H:%M:%S"); // or just %T in this case
		int end_date = tm_end.tm_year * 10000 + tm_end.tm_mon * 100 + tm_end.tm_mday;
		int end_seconds = tm_end.tm_sec + tm_end.tm_min * 60 + tm_end.tm_hour * 3600;

		if (!transportation_modes_table.contains(token[2])) 
		{
			//TODO: new transportation mode
			int br = 3;
		};

		TransportationMode mode = transportation_modes_table.at(token[2]);

		// Activity is during midnight -> make two records
		if (start_date != end_date)
		{
			labels.Add(start_date, start_seconds, 86400, mode);
			labels.Add(end_date, 0, end_seconds, mode);
		} else
		{
			labels.Add(start_date, start_seconds, end_seconds, mode);
		}
	}

	file.close();

	return labels;
}