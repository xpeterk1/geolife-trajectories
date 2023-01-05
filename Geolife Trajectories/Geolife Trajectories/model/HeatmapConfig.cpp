#include "HeatmapConfig.h"

int HeatmapConfig::GetMode() 
{
	int mode = 0;
	if (walk) mode |= WALK;
	if (bike) mode |= BIKE;
	if (bus) mode |= BUS;
	if (car) mode |= CAR;
	if (subway) mode |= SUBWAY;
	if (train) mode |= TRAIN;
	if (airplane) mode |= AIRPLANE;
	if (boat) mode |= BOAT;
	if (run) mode |= RUN;
	if (motorcycle) mode |= MOTORCYCLE;

	// Everything is checked -> include unknown values
	if (mode == 2046) mode = 2047;

	return mode;
} 

bool HeatmapConfig::NeedsRecomputation(int* newMode)
{
	// Recomputation is needed when current states of bools do not match previous mode
	int currentMode = GetMode();

	if (currentMode != last_mode)
	{
		last_mode = currentMode;
		*newMode = currentMode;
		return true;
	}


	return false;
}