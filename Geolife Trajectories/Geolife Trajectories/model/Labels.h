#pragma once

#include <map>
#include "Datapoint.h"
#include <vector>

struct Label
{
public:
	int start_seconds;
	int end_seconds;
	TransportationMode mode;
};

class Labels
{
public:

	// Date -> (Start, End, Type)
	std::map<int, std::vector<Label>> labels;
private:

public:
	void Add(int date, int start, int end, TransportationMode mode);
	bool Empty();
	void Clear();

private:

};

