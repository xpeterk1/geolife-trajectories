#pragma once
#include <ctime>
#include <fstream>

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
	Datapoint();
	Datapoint(float x, float y, TransportationMode mode, time_t time);

	// Serialize the object to a stream
	friend std::ostream& operator<<(std::ostream& out, const Datapoint& p) {
		out << p.x << ' ' << p.y << ' ' << static_cast<int>(p.mode) << ' ' << p.time;
		return out;
	}

	// Deserialize the object from a stream
	friend std::istream& operator>>(std::istream& in, Datapoint& p) {
		int m;
		in >> p.x >> p.y >> m >> p.time;
		p.mode = static_cast<TransportationMode>(m);
		return in;
	}

private:

};

