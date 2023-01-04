#pragma once
#include <ctime>

enum TransportationMode : int
{
	UNKNOWN, WALK, BIKE, BUS, CAR, TRAIN, AIRPLANE, OTHER
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

