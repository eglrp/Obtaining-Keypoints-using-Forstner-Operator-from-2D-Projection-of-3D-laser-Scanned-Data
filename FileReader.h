//Class FileReader is defined in this header file where the functions for Reading Valuess, Line and Polygon from
//given text files has been declared. 
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "Values.h"


using namespace std;

class FileReader
{
public:	
	FileReader(); 
	~FileReader(); 

	vector<Values*> ReadFile(const string path);
	

	//vector<string> Split(const string &s, const string &delim);
};

FileReader::FileReader()
{}

FileReader:: ~FileReader()
{}

//Function for reading Valuess from file has been declared
vector<Values*> FileReader::ReadFile(string path)
{
	vector<Values*> result; //container for Values Valuesers
	double x, y, z, inten, r, g, b; //Values coordinates
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
		file >> x >> y >> z >> inten >> r >> g >> b; //read in double x and double y
						//cout << x << " "<< y << endl; //line not needed - only to check if reading was ok
		result.push_back(new Values(x, y, z, inten, r, g, b)); //create a Values and put it into a container
	}
	result.pop_back(); //the last line is readed twice, so you must delete the last Values

	return result;
}