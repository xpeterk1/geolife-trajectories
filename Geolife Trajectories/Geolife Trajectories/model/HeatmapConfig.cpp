#include "HeatmapConfig.h"

bool HeatmapConfig::NeedsRecomputation()
{
	// Recomputation is needed when current states of bools do not match previous mode
	return current_mode != last_mode;
}

void HeatmapConfig::Switch(TransportationMode mode)
{
	current_mode ^= mode;
}