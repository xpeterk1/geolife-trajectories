#include "Datapoint.h"

Datapoint::Datapoint(float x, float y, TransportationMode mode, time_t time)
{
	this->x = x;
	this->y = y;
	this->mode = mode;
	this->time = time;
}