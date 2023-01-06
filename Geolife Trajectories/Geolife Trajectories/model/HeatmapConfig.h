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

private:
	int last_mode = 2047;
	int current_mode = 0;

public:
	void Switch(TransportationMode mode);
	bool NeedsRecomputation(int* newMode);

private:

};

