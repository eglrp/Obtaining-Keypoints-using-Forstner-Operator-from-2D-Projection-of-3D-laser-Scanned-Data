#pragma once
using namespace std;

class Values
{
public:
	double x, y,z,inten,r,g,b;


	Values(double my_x, double my_y,double my_z,double my_inten, double my_r,double my_g, double my_b);   //constructor
	Values(); // again, different contructor - empty member variables
	~Values(); //destructor

};

Values::Values(double my_x, double my_y, double my_z, double my_inten, double my_r, double my_g, double my_b)
	:x(my_x), y(my_y), z(my_z), inten(my_inten), r(my_r), g(my_g), b(my_b)
{}

Values::Values()
{}
Values:: ~Values()
{}


