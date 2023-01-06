#include "HeatmapConfig.h"

bool HeatmapConfig::NeedsRecomputation(int* newMode)
{
	// Recomputation is needed when current states of bools do not match previous mode

	if (current_mode != last_mode)
	{
		last_mode = current_mode;
		*newMode = current_mode;
		return true;
	}

	return false;
}

void HeatmapConfig::Switch(TransportationMode mode)
{
	current_mode ^= mode;
}