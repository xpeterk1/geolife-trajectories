#pragma once

#include "../model/Datapoint.h"

class HeatmapConfig
{
public:
	bool unknown = false;
	bool walk = false;
	bool bike = false;
	bool bus = false;
	bool car = false;
	bool subway = false;
	bool train = false;
	bool airplane = false;
	bool boat = false;
	bool run = false;
	bool motorcycle = false;
	bool use_log_scale = true;

	bool time_changed = false;
	bool scaling_changed = false;
	bool size_changed = false;
	bool std_changed = false;

	int min_time = 0; // 00:00:00
	int max_time = 86400; // 24:00:00

	int kernel_size = 5;
	float sigma = 1.0f;

	int current_mode = 0;
	int last_mode = 2047;

private:

public:
	void Switch(TransportationMode mode);
	bool NeedsRecomputation();
	bool ReuploadKernel();

private:

};

