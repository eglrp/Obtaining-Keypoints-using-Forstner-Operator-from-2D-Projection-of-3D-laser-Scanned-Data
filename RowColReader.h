#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "RowCol.h"


using namespace std;

class RowColReader
{
public:
	RowColReader();
	~RowColReader();

	vector<RowCol*> ReadRowColsFile(const string path);
};

RowColReader::RowColReader()
{}

RowColReader:: ~RowColReader()
{}

//Function for reading RowCols from file has been declared
vector<RowCol*> RowColReader::ReadRowColsFile(string path)
{
	vector<RowCol*> rowcol; //container for RowCol RowColers
	double row, col ; //RowCol coordinates
	ifstream file(path, ios::in); //open file with the given path, to read-in only
	
	while (file && !file.eof()) //if such a file exists and till it's not the end-of-file
	{
		
		file >> col >> row; //read in double x and double y
		file.close();
		rowcol.push_back(new RowCol(col, row)); //create a RowCol and put it into a container
	}
	
	rowcol.pop_back(); 

	return rowcol;
}