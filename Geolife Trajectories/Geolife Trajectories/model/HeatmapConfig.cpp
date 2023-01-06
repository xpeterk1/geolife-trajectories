#include "HeatmapConfig.h"

bool HeatmapConfig::NeedsRecomputation()
{
	// Recomputation is needed when current states of bools do not match previous mode
	return current_mode != last_mode || time_changed || scaling_changed || size_changed || std_changed;
}

bool HeatmapConfig::ReuploadKernel() 
{
	return size_changed || std_changed;
}

void HeatmapConfig::Switch(TransportationMode mode)
{
	current_mode ^= mode;
}