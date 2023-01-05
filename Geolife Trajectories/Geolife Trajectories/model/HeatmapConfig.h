#pragma once

#include "../model/Datapoint.h"

class HeatmapConfig
{
public:
	bool walk = true;
	bool bike = true;
	bool bus = true;
	bool car = true;
	bool subway = true;
	bool train = true;
	bool airplane = true;
	bool boat = true;
	bool run = true;
	bool motorcycle = true;

private:
	// start with all modes enabled
	int last_mode = 0;

public:
	int GetMode();
	bool NeedsRecomputation(int* newMode);

private:

};

