//Class GrayFileReader is defined in this header file where the functions for Reading GrayValuess, Line and Polygon from
//given text files has been declared. 
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "GrayValues.h"


using namespace std;

class GrayFileReader
{
public:	
	GrayFileReader(); 
	~GrayFileReader(); 

	vector<GrayValues*> ReadGrayFile(const string path);
	

	//vector<string> Split(const string &s, const string &delim);
};

GrayFileReader::GrayFileReader()
{}

GrayFileReader:: ~GrayFileReader()
{}

//Function for reading GrayValuess from file has been declared
vector<GrayValues*> GrayFileReader::ReadGrayFile(string path)
{
	vector<GrayValues*> result_gray; //container for GrayValues GrayValuesers
	double x_g, y_g, z_g, inten_g; //GrayValues coordinates
	string waste_line;
	ifstream file(path, ios::in); //open file with the given path, to read-in only
	//Wasting first 10 line
	{
		getline(file, waste_line);getline(file, waste_line);getline(file, waste_line);getline(file, waste_line);
		getline(file, waste_line);getline(file, waste_line);getline(file, waste_line);getline(file, waste_line);
		getline(file, waste_line);getline(file, waste_line);
	}
	while (file && !file.eof()) //if such a file exists and till it's not the end-of-file
	{
		//file.ignore(100, '(');   // ignore until bracket
		file >> x_g >> y_g >> z_g >> inten_g; //read in double x and double y
						//cout << x << " "<< y << endl; //line not needed - only to check if reading was ok
		result_gray.push_back(new GrayValues(x_g, y_g, z_g, inten_g)); //create a GrayValues and put it into a container
	}
	result_gray.pop_back(); //the last line is readed twice, so you must delete the last GrayValues

	return result_gray;
}