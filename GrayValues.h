#pragma once
using namespace std;

class GrayValues
{
public:
	double x, y,z,inten;


	GrayValues(double my_x, double my_y,double my_z,double my_inten);   //constructor
	GrayValues(); // again, different contructor - empty member variables
	~GrayValues(); //destructor

};


GrayValues::GrayValues(double my_x, double my_y, double my_z, double my_inten)
	:x(my_x), y(my_y), z(my_z), inten(my_inten)
{}

GrayValues::GrayValues()
{}
GrayValues:: ~GrayValues()
{}


