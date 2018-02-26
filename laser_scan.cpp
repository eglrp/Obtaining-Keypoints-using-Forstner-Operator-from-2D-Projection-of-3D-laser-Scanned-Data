//============================================================================
// Name        : Laser Scanning Assignment
// Author      : Arghadeep Mazumder
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2\core\core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include"Values.h"
#include"FileReader.h"
#include"RowColReader.h"
#include"GrayValues.h"
#include"GrayFileReader.h"
#include<math.h>
#include<string.h>
using namespace std;

/***************** INTENSITY TO IMAGE *********************/
cv::Mat inten_to_img(vector<Values*> points, cv::Mat img) {
	int row = img.rows;
	int col = img.cols;
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			img.at<float>(i, j) = points[row*j + i]->inten*(255.0);
		}
	}
	return img;
}

cv::Mat inten_to_img_gray(vector<GrayValues*> points, cv::Mat img) {
	int row = img.rows;
	int col = img.cols;
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			img.at<float>(i, j) = points[row*j + i]->inten*(255.0);
		}
	}
	return img;
}
/***************** RGB VALUES TO IMAGE ********************/
//cv::Mat rgb_to_img() {
//
//}

/********************* HISTOGRAM: RGB *********************/
cv::Mat histplotrgb(cv::Mat RGB_Image) {
	cv::Mat channel[3];
	cv::Mat b_hist;
	cv::Mat g_hist;
	cv::Mat r_hist;
	int hist_size = 256;
	float range[] = { 0,256 };
	const float* hist_range = { range };
	cv::split(RGB_Image, channel);
	/*Histogram for RBG */
	cv::calcHist(&channel[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, &hist_range, true, false);
	cv::calcHist(&channel[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, &hist_range, true, false);
	cv::calcHist(&channel[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, &hist_range, true, false);
	//Drawing histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / hist_size);
	cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	//Normalizing Histogram
	cv::normalize(b_hist, b_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	//Drawing for each channel
	for (int i = 1;i < hist_size;i++) {
		cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 0, 0, 0);
		cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 0, 0, 0);
		cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 0, 0, 0);
	}
	return hist_image;
}

/********************* HISTOGRAM: GREY SCALE *********************/
cv::Mat histplotgrey(cv::Mat greyscale_image) {
	cv::Mat inten_hist;
	int hist_size = 256;
	float range[] = { 0,256 };
	const float* hist_range = { range };
	cv::calcHist(&greyscale_image, 1, 0, cv::Mat(), inten_hist, 1, &hist_size, &hist_range, true, false);
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / hist_size);
	cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat hist_inten_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(inten_hist, inten_hist, 0, hist_inten_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1;i < hist_size;i++) {
	cv::line(hist_inten_image, cv::Point(bin_w*(i - 1), hist_h - cvRound(inten_hist.at<float>(i - 1))),
	cv::Point(bin_w*(i), hist_h - cvRound(inten_hist.at<float>(i))),
	cv::Scalar(255, 255, 255), 0, 0, 0);
	}
	return hist_inten_image;
}

/********************* MINIMUM INTENSITY *********************/
int minimum(cv::Mat img, int rows,int cols) {
	float min = img.at<float>(0, 0);
	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < cols;j++) {
			if (img.at<float>(i, j) < min) {
				min = img.at<float>(i, j);
			}
		}
	}
	return min;
}

/********************* MAXIMUM INTENSITY *********************/
int maximum(cv::Mat img, int rows, int cols) {
	float max = img.at<float>(0, 0);
	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < cols;j++) {
			if (img.at<float>(i, j) > max) {
				max = img.at<float>(i, j);
			}
		}
	}
	return max;
}

/*************** Intensity Values as .txt file *****************/
void print_inten(cv::Mat any_image) {
	int row = any_image.rows;
	int col = any_image.cols;
	ofstream opfile;
	opfile.open("code.txt");
	for (int i = 0;i < row;i++) {
	for (int j = 0;j < col;j++) {
	opfile << any_image.at<float>(i, j) << " ";
	}
	opfile<<endl;
	}
	opfile.close();
}
/*************** Contrast Enhancement *****************/
cv::Mat contrast_enahncement(cv::Mat image, int min, int max) {
	cv::Mat img_grey_scale = image.clone();
	int row = img_grey_scale.rows;
	int col = img_grey_scale.cols;
	for (int k = 0;k < row;k++) {
		for (int l = 0;l < col;l++) {
			img_grey_scale.at<float>(k, l) = ((image.at<float>(k, l) - min) / (max - min)) * 255;
		}
	}
	return img_grey_scale;
}

cv::Mat rotation_img(cv::Mat src,int angle) {
	cv::Mat img_rgb_rot;
	/*cout << "Enter the Rotation Angle: " << endl << endl;
	cout << "Enter 0 : No Rotation [Recommended] " << endl;
	cout << "Enter 90: Left Rotation" << endl;
	cout << "Enter 190: Bottom Up Rotation" << endl;
	cout << "Enter 270: Right Rotation" << endl << endl;
	cin >> angle;*/
	cv::Point2f pt(src.cols / 2, src.rows / 2);
	cv::Mat pr = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::warpAffine(src, img_rgb_rot, pr, cv::Size(src.cols, src.rows));
	return img_rgb_rot;
}

/******************** Forstner Corner Detector *********************/
/***************************** Start *******************************/
cv::Mat kernel_der(double sigma) {
	int ksize;
	cout << "Enter the size of the Kernel : " << endl << endl;
	cout << "[Hints: 3, 5(Recommended), 7]" << endl;
	cin >> ksize;
	/*cout << endl << "Enter the Value of Sigma : " << endl;
	cin >> sigma; cout << endl;*/
	cv::Mat X_kernel = cv::getGaussianKernel(ksize, sigma, CV_32FC1);
	cv::Mat Y_kernel = cv::getGaussianKernel(ksize, sigma, CV_32FC1);
	cv::Mat kernel = cv::Mat::ones(ksize, ksize, CV_32FC1);
	cv::Mat XY_kernel = X_kernel*Y_kernel.t();
	for (int i = 0;i < ksize;i++) {
		for (int j = 0;j < ksize;j++) {
			int ru = i - ksize / 2;
			XY_kernel.at<float>(i, j) = -ru*XY_kernel.at<float>(i, j) / (pow(sigma, 2));
		}
	}
	return XY_kernel;
}

//Gx.Gx , Gx.Gy, Gy.Gy
cv::Mat mul_gradXY(cv::Mat gradX, cv::Mat gradY) {

	cv::Mat mul_grad = cv::Mat::zeros(gradX.size(), CV_32FC1);
	for (int i = 0;i < gradX.rows;i++) {
		for (int j = 0;j < gradX.cols;j++) {
			mul_grad.at<float>(i, j) = gradX.at<float>(i, j)*gradY.at<float>(i, j);
		}
	}
	cv::GaussianBlur(mul_grad, mul_grad, cv::Size(3, 3), 0.5, 0.5);
	return mul_grad;
}
//Trace of Structure Window
cv::Mat trace(cv::Mat gradX, cv::Mat gradY) {
	cv::Mat trace_XY = cv::Mat::zeros(gradX.size(), CV_32FC1);
	for (int i = 0;i < gradX.rows;i++) {
		for (int j = 0;j < gradX.cols;j++) {
			trace_XY.at<float>(i, j) = gradX.at<float>(i, j) + gradY.at<float>(i, j);
		}
	}
	return trace_XY;
}
//Determinant of Structure Tensor
cv::Mat det_tensor(cv::Mat gradX, cv::Mat gradY, cv::Mat grad_XY) {
	cv::Mat determinant_mat = gradX.clone();
	for (int i = 0;i < gradX.rows;i++) {
		for (int j = 0;j < gradY.cols;j++) {
			determinant_mat.at<float>(i, j) = (gradX.at<float>(i, j)*gradY.at<float>(i, j)) - (grad_XY.at<float>(i, j)*grad_XY.at<float>(i, j));
		}
	}
	return determinant_mat;
}
//Weight Calculation
cv::Mat weight(cv::Mat det, cv::Mat trace) {
	cv::Mat weight_mat = det.clone();
	for (int i = 0;i < det.rows;i++) {
		for (int j = 0;j < det.cols;j++) {
			if (trace.at<float>(i, j) != 0) {
				weight_mat.at<float>(i, j) = det.at<float>(i, j) / trace.at<float>(i, j);
			}
		}
	}
	return weight_mat;
}
// Non-Max Suppression
cv::Mat nonMaxSuppression(cv::Mat& img) {

	cv::Mat out = img.clone();

	for (int x = 1; x<out.cols - 1; x++) {
		for (int y = 1; y<out.rows - 1; y++) {
			if (img.at<float>(y - 1, x) > img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y, x - 1) > img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y, x + 1) > img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
			if (img.at<float>(y + 1, x) > img.at<float>(y, x)) {
				out.at<float>(y, x) = 0;
				continue;
			}
		}
	}
	return out;
}
//Isotropy
cv::Mat isotropy_img(cv::Mat det, cv::Mat trace) {
	cv::Mat isotropy_mat = det.clone();
	for (int i = 0;i < det.rows;i++) {
		for (int j = 0;j < det.cols;j++) {
			isotropy_mat.at<float>(i, j) = (4 * det.at<float>(i, j)) / (trace.at<float>(i, j)*trace.at<float>(i, j));
		}
	}
	return isotropy_mat;
}
//Mean Calculation
float mean(cv::Mat src) {
	float mean_weight = 0.0;
	for (int i = 0;i < src.rows;i++) {
		for (int j = 0;j < src.cols;j++) {
			mean_weight += src.at<float>(i, j);
		}
	}
	mean_weight = (mean_weight / (src.rows*src.cols));
	cout << "Mean Weight: " << mean_weight << endl << endl;
	return mean_weight;
}
//Keypoint Generation
vector<cv::KeyPoint> key_point(cv::Mat weight_ST, cv::Mat isotropy_ST, float weight_mean) {
	vector<cv::KeyPoint> points;
	for (int i = 0;i < weight_ST.rows;i++) {
		for (int j = 0;j < weight_ST.cols;j++) {
			if (weight_ST.at<float>(i, j) > weight_mean*0.5 && weight_ST.at<float>(i, j) < weight_mean*1.5 && isotropy_ST.at<float>(i, j) > 0.5 && isotropy_ST.at<float>(i, j) < 0.75) {
				points.push_back(cv::KeyPoint(i, j, 0.1));
			}
		}
	}
	cout << "Number of Detected Points : " << points.size() << endl << endl;
	return points;
}
//Keypoint Extraction for x,y,z,inten,r,g,b
vector<Values*> keypoint_extract(vector<cv::KeyPoint> points, vector<Values*> point_key,cv::Mat src) {
	vector<Values*> new_points;
	double n_x, n_y, n_z, n_inten, n_r, n_g, n_b;
	for (int i = 0;i < points.size();i++) {
		new_points.push_back(point_key[points[i].pt.x*src.cols + points[i].pt.y]);
	}
	cout << new_points.size() << endl;
	return new_points;
}
//Keypoint Extraction for x,y,z,inten,r,g,b
vector<GrayValues*> keypoint_extract_xyzinten(vector<cv::KeyPoint> points, vector<GrayValues*> point_key, cv::Mat src) {
	vector<GrayValues*> new_points;
	double n_x, n_y, n_z, n_inten, n_r, n_g, n_b;
	for (int i = 0;i < points.size();i++) {
		new_points.push_back(point_key[points[i].pt.x*src.cols + points[i].pt.y]);
	}
	cout << new_points.size() << endl;
	return new_points;
}
// Generating Keypoints in txt file for x,y,z,inten,r,g,b
void print_newpoints(vector<Values*> newpoints) {
	ofstream opfile;
	string input = "";
	cout << "Enter the Name of the file: " << endl;
	cin >> input;
	opfile.open(string(input)+string(".txt").c_str());
	for (int i = 0; i < newpoints.size(); i++) {
		opfile << newpoints[i]->x << " " << newpoints[i]->y << " " << newpoints[i]->z << " " <<
			newpoints[i]->inten << " " << newpoints[i]->r << " " << newpoints[i]->g << " " << newpoints[i]->b << endl;
	}
	opfile.close();
}
// Generating Keypoints in txt file for x,y,z,inten
void print_newpoints_xyzinten(vector<GrayValues*> newpoints) {
	ofstream opfile;
	string input = "";
	cout << "Enter a name to save the file [Without any extension]: " << endl;
	cin >> input;
	opfile.open(string(input) + string(".txt").c_str());
	for (int i = 0; i < newpoints.size(); i++) {
		opfile << newpoints[i]->x << " " << newpoints[i]->y << " " << newpoints[i]->z << " " <<	newpoints[i]->inten << " " << endl;
	}
	opfile.close();
	cout << "Keyppoints Extracted" << endl << endl;
}
//Drawing Keypoints
cv::Mat draw_points(cv::Mat src, vector<cv::KeyPoint> point) {
	cv::Mat img = src.clone();
	drawKeypoints(src, point, img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	return img;
}
//Forstner Keypoint Detection for x,y,z,inten,r,g,b 
cv::Mat forstner(cv::Mat src, cv::Mat op, vector<Values*> points_of_interest) {
	float weight_mean;
	cv::Mat xder_kernel, yder_kernel;
	cv::Mat deriX, deriY; //Derivative of X & Y
	cv::Mat fxfx, fyfy, fxfy;//Elements of Structure Tensor
	cv::Mat determinant_ST; //Determinant
	cv::Mat trace_ST = src.clone(); //Trace of Structure Tensor
	cv::Mat weight_ST; //Weight of Structure Tensor
	cv::Mat isotropy_ST; //Isotropy of Structure Tensor
	cv::Mat output;
	vector<cv::KeyPoint> points;
	vector<Values*> new_points;

	//Gradient of Image in X and Y dir using Gaussian Kernel
	xder_kernel = kernel_der(0.25);
	yder_kernel = xder_kernel.t();
	filter2D(src, deriX, CV_32FC1, xder_kernel);
	filter2D(src, deriY, CV_32FC1, yder_kernel);
	//Creating Structure Tensor Elements
	fxfx = mul_gradXY(deriX, deriX);
	fyfy = mul_gradXY(deriY, deriY);
	fxfy = mul_gradXY(deriX, deriY);
	//Computing Trace
	trace_ST = trace(fxfx, fyfy);
	//Computing Determinant
	determinant_ST = det_tensor(fxfx, fyfy, fxfy);
	//Computing Weight
	weight_ST = weight(determinant_ST, trace_ST);
	nonMaxSuppression(weight_ST);
	//Isotropy
	isotropy_ST = isotropy_img(determinant_ST, trace_ST);
	nonMaxSuppression(isotropy_ST);
	//Computing Mean Weight 
	weight_mean = mean(weight_ST);
	//Point Computation
	points = key_point(weight_ST, isotropy_ST, weight_mean);
	//Draw points
	output = draw_points(op, points);
	//Extracting keypoints
	new_points = keypoint_extract(points, points_of_interest, src);
	//Merging Keypoints to the txt file
	print_newpoints(new_points);
	return output;
}
//Forstner Keypoint Detection for x,y,z,inten
cv::Mat forstner_xyzinten(cv::Mat src, cv::Mat op, vector<GrayValues*> points_of_interest) {
	float weight_mean;
	cv::Mat xder_kernel, yder_kernel;
	cv::Mat deriX, deriY; //Derivative of X & Y
	cv::Mat fxfx, fyfy, fxfy;//Elements of Structure Tensor
	cv::Mat determinant_ST; //Determinant
	cv::Mat trace_ST = src.clone(); //Trace of Structure Tensor
	cv::Mat weight_ST; //Weight of Structure Tensor
	cv::Mat isotropy_ST; //Isotropy of Structure Tensor
	cv::Mat output;
	vector<cv::KeyPoint> points;
	vector<GrayValues*> new_points;

	//Gradient of Image in X and Y dir using Gaussian Kernel
	xder_kernel = kernel_der(0.25);
	yder_kernel = xder_kernel.t();
	filter2D(src, deriX, CV_32FC1, xder_kernel);
	filter2D(src, deriY, CV_32FC1, yder_kernel);
	//Creating Structure Tensor Elements
	fxfx = mul_gradXY(deriX, deriX);
	fyfy = mul_gradXY(deriY, deriY);
	fxfy = mul_gradXY(deriX, deriY);
	//Computing Trace
	trace_ST = trace(fxfx, fyfy);
	//Computing Determinant
	determinant_ST = det_tensor(fxfx, fyfy, fxfy);
	//Computing Weight
	weight_ST = weight(determinant_ST, trace_ST);
	nonMaxSuppression(weight_ST);
	//Isotropy
	isotropy_ST = isotropy_img(determinant_ST, trace_ST);
	nonMaxSuppression(isotropy_ST);
	//Computing Mean Weight 
	weight_mean = mean(weight_ST);
	//Point Computation
	points = key_point(weight_ST, isotropy_ST, weight_mean);
	//Draw points
	output = draw_points(op, points);
	//Extracting keypoints
	new_points = keypoint_extract_xyzinten(points, points_of_interest, src);
	//Merging Keypoints to the txt file
	print_newpoints_xyzinten(new_points);
	return output;
}
/***************************** End *******************************/

int main(int argc, char** argv) {
	FileReader file_read;
	RowColReader rowcol_read;
	GrayFileReader gray_read;
	vector<GrayValues*> gray_val;
	vector<RowCol*> rowcol;
	vector<Values*> points_of_interest;

	int row, col, file_num, num;
	cout << "\t*************  Laser Scanning Assignment: 2  *************" << endl << endl;
	cout << "Please Select a File to Proceed" << endl << endl;
	cout << "TU Main Building:       Select 1 " << endl;
	cout << "Sixth Floor [Room 504]: Select 2 " << endl;
	cout << "Sixth Floor [Room 505]: Select 3 " << endl;
	cout << "Orangerie:              Select 4 " << endl;
	cout << "TangXianzong:           Select 5 " << endl<<endl;
	cin >> file_num;
	if (file_num == 1 || file_num == 2 || file_num == 3 || file_num == 4 || file_num == 5) {
		num = file_num;	
	}
	else {
		cout << "Wrong Entry!!!" << endl;
		cin.get();
		system("pause");
		return -1;
	}
	if (num == 1 || num == 4 || num == 5) {
		cout << "Point File :  Loading" << endl << endl;
		rowcol = rowcol_read.ReadRowColsFile(argv[num]);
		cout << "Point File :  Done" << endl << endl;
		//Assigning Image Row Column 
		for (int i = 0;i < rowcol.size();i++) {
			row = rowcol[i]->r;
			col = rowcol[i]->c;
		}
		cout << "Number of Rows:    " << row << endl;
		cout << "Number of Columns: " << col << endl << endl;
		//Generation of blank images 
		cv::Mat channel[3];
		cv::Mat img = cv::Mat::zeros(row, col, CV_32FC1);
		cv::Mat img_rgb = cv::Mat::zeros(row, col, CV_8UC3);
		//Reading the ptx file and inserting the values in a vector
		/************* Intensity to Image *************/
		cout << "\t**************  Greyscale Image Operation **************" << endl << endl;
		cout << "Greyscale Image :  Generating (Please Wait)" << endl << endl;
		points_of_interest = file_read.ReadFile(argv[num]);
		img = inten_to_img(points_of_interest, img);
		cv::imwrite("GreyScale_Image.png", img);
		cout << "Greyscale Image :  Done" << endl << endl;
		/************* Min & Max Intensity *************/
		int min_inten = minimum(img, row, col);
		int max_inten = maximum(img, row, col);
		cout << "Greyscale Minimum Intenstiy: " << min_inten / 255.0 << endl;
		cout << "Greyscale Maximum Intenstiy: " << max_inten / 255.0 << endl << endl;
		cout << "Generating Histogram" << endl;
		cv::Mat img_grey_hist = histplotgrey(img);
		cv::imwrite("GreyScale_Image_Histogram.png", img_grey_hist);
		cout << "Histogram Generated" << endl << endl;
		cout << "Contrast Enhancement:  Processing" << endl;
		cv::Mat img_grey_contrast = contrast_enahncement(img, min_inten, max_inten);
		cv::imwrite("GreyScale_Image_EnhanCont.png", img_grey_contrast);
		cout << "Contrast Enhancement:  Done" << endl << endl;
		cout << "Generating Histogram : Enhanced Contrast " << endl;
		cv::Mat img_grey_hist_contrast = histplotgrey(img_grey_contrast);
		cv::imwrite("GreyScale_Image_EnhanCont_Histogram.png", img_grey_hist_contrast);
		cout << "Histogram Generated" << endl << endl;
		/***************** RGB Image ******************/
		cout << "\t**************  RGB Image Operation **************" << endl << endl;
		cout << "RGB Image : Generating (Please Wait)" << endl << endl;
		cv::split(img_rgb, channel);
		for (int i = 0;i < row;i++) {
			for (int j = 0; j < col;j++) {
				channel[0].at<uchar>(i, j) = points_of_interest[row*j + i]->b;
				channel[1].at<uchar>(i, j) = points_of_interest[row*j + i]->g;
				channel[2].at<uchar>(i, j) = points_of_interest[row*j + i]->r;
			}
		}
		cv::merge(channel, 3, img_rgb);
		img_rgb.convertTo(img_rgb, CV_32FC1);
		cv::imwrite("RGB_Image.png", img_rgb);
		cout << "RGB Image : Done" << endl << endl;
		cout << "Generating Histogram : RGB " << endl;
		cv::Mat img_rgb_hist = histplotrgb(img_rgb);
		cv::imwrite("RGB_Image_Histogram.png", img_rgb_hist);
		cout << "Histogram Generated" << endl << endl;
		cout << "\t**************  Corner Detection **************" << endl << endl;
		cout << "Corner Detection Grey Scale:  Processing" << endl;
		cv::Mat forstner_corner_detection = forstner(img, img.t(),points_of_interest);
		cv::imwrite("Forstner_Corner_Grey_Scale.png", forstner_corner_detection);
		cout << "Corner Detection Grey Scale:  Done" << endl << endl;
		cout << "Corner Detection RGB:  Processing" << endl;
		cv::Mat img_rgb_rot;
		cv::flip(img_rgb, img_rgb_rot, 1);
		cv::Mat forstner_corner_detection_rgb = forstner(img_rgb, img_rgb_rot.t(),points_of_interest);
		cv::imwrite("Forstner_Corner_RGB.png", forstner_corner_detection_rgb);
		cout << "Corner Detection RGB Scale:  Done" << endl;
	}
	else {
		cout << "Point File :  Loading" << endl << endl;
		rowcol = rowcol_read.ReadRowColsFile(argv[num]);
		cout << "Point File :  Done" << endl << endl;
		//Assigning Image Row Column 
		for (int i = 0;i < rowcol.size();i++) {
			row = rowcol[i]->r;
			col = rowcol[i]->c;
		}
		cout << "Number of Rows: " << row << endl;
		cout << "Number of Columns: " << col << endl << endl;
		cv::Mat img = cv::Mat::zeros(row, col, CV_32FC1);
		cout << "\t**************  Greyscale Image Operation **************" << endl << endl;
		cout << "Greyscale Image :  Generating (Please Wait)" << endl << endl;
		gray_val = gray_read.ReadGrayFile(argv[num]);
		img = inten_to_img_gray(gray_val, img);
		cv::imwrite("GreyScale_Image.png", img);
		cout << "Greyscale Image :  Done" << endl << endl;
		/************* Min & Max Intensity *************/
		int min_inten = minimum(img, row, col);
		int max_inten = maximum(img, row, col);
		cout << "Greyscale Minimum Intenstiy: " << min_inten / 255.0 << endl;
		cout << "Greyscale Maximum Intenstiy: " << max_inten / 255.0 << endl << endl;
		cout << "Generating Histogram" << endl;
		cv::Mat img_grey_hist = histplotgrey(img);
		cv::imwrite("GreyScale_Image_Histogram.png", img_grey_hist);
		cout << "Histogram Generated" << endl << endl;
		cout << "Contrast Enhancement:  Processing" << endl;
		cv::Mat img_grey_contrast = contrast_enahncement(img, min_inten, max_inten);
		cv::imwrite("GreyScale_Image_EnhanCont.png", img_grey_contrast);
		cout << "Contrast Enhancement:  Done" << endl << endl;
		cout << "Generating Histogram : Enhanced Contrast " << endl;
		cv::Mat img_grey_hist_contrast = histplotgrey(img_grey_contrast);
		cv::imwrite("GreyScale_Image_EnhanCont_Histogram.png", img_grey_hist_contrast);
		cout << "Histogram Generated" << endl << endl;
		cout << "\t**************  Corner Detection **************" << endl << endl;
		cout << "Corner Detection Grey Scale:  Processing" << endl;
		cv::Mat forstner_corner_detection = forstner_xyzinten(img, img.t(),gray_val);
		cv::imwrite("Forstner_Corner_Grey_Scale.png", forstner_corner_detection);
		cout << "Corner Detection Grey Scale:  Done" << endl << endl;
	}
	system("pause");
	return 0;
}