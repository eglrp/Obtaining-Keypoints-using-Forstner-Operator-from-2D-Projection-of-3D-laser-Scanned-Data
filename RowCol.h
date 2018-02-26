#pragma once
using namespace std;

class RowCol
{
public:
	double r, c;


	RowCol(double my_c, double my_r);   //constructor
	RowCol(); // again, different contructor - empty member variables
	~RowCol(); //destructor

};

RowCol::RowCol(double my_c, double my_r)
	:c(my_c), r(my_r)
{}

RowCol::RowCol()
{}
RowCol:: ~RowCol()
{}