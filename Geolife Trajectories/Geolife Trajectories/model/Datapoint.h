#pragma once
#include <ctime>
#include <fstream>

enum TransportationMode : int
{
	UNKNOWN = 0b1,
	WALK = 0b10,
	BIKE = 0b100,
	BUS = 0b1000,
	CAR = 0b10000,
	SUBWAY = 0b100000,
	TRAIN = 0b1000000,
	AIRPLANE = 0b10000000,
	BOAT = 0b100000000,
	RUN = 0b1000000000,
	MOTORCYCLE = 0b10000000000
};

class Datapoint
{
public:
	float x;
	float y;
	TransportationMode mode;
	int time;

private:

public:
	Datapoint();
	Datapoint(float x, float y, TransportationMode mode, time_t time);

	// Serialize the object to a stream
	friend std::ostream& operator<<(std::ostream& out, const Datapoint& p) {
		out << p.x << p.y << static_cast<int>(p.mode) << p.time;
		return out;
	}

	// Deserialize the object from a stream
	friend std::istream& operator>>(std::istream& in, Datapoint& p) {
		int m;
		in >> p.x;
		in >> p.y;
		in >> m;
		in >> p.time;
		p.mode = static_cast<TransportationMode>(m);
		return in;
	}

private:

};

