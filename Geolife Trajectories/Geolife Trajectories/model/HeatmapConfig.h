#pragma once

#include "../model/Datapoint.h"

class HeatmapConfig
{
public:
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
	// start with all modes enabled
	int last_mode = 2047;

public:
	int GetMode();
	bool NeedsRecomputation(int* newMode);

private:

};

