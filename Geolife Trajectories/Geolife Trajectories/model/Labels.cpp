#include "Labels.h"

void Labels::Add(int date, int start, int end, TransportationMode mode)
{
	if (labels.contains(date))
	{
		labels[date].push_back(Label{ start, end, mode });
	}
	else
	{
		std::vector<Label> l { Label{ start, end, mode } };
		labels.insert({date, l});
	}
}

bool Labels::Empty()
{
	return labels.size() == 0;
}

void Labels::Clear()
{
	labels.clear();
}