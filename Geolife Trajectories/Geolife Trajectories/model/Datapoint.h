#pragma once
#include <ctime>

enum TransportationMode : int
{
	UNKNOWN, WALK, BIKE, BUS, CAR, SUBWAY, TRAIN, AIRPLANE, BOAT, RUN, MOTORCYCLE
};

class Datapoint
{
public:
	float x;
	float y;
	TransportationMode mode;
	time_t time;

private:

public:
	Datapoint(float x, float y, TransportationMode mode, time_t time);

private:

};

