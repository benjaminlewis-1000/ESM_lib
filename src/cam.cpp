// aaa.c

#include <stdio.h>
#include <stdlib.h>

#if defined (__cplusplus)
extern "C"{
#endif
	#include "ESMlibry.h"  
	// Prevent name mangling, which happens when you link
	// a C library into a C++ program.
#if defined (__cplusplus)
}
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	
	
 	 cv::VideoCapture cap(0);
	
 /*	Mat M = imread( "test.jpg" );
 	
 	cvtColor(M, M, CV_BGR2GRAY);
 	
 	imshow("Test",M);
 	waitKey(0);
 	
 	imageStruct I;
 	
 	MallImage(&I, M.cols, M.rows);
 	for (int i = 0; i < M.rows; i++){  // Across image
 		for (int j = 0; j < M.cols; j++){ // Down image
 			I.data[i*M.cols + j] = M.at<uchar>(i,j);
 		}
 	}*/
 	
 	namedWindow("view");
 	
 	Mat KF;
	cap >> KF;
	cvtColor(KF, KF, CV_BGR2GRAY);
	
	imshow("view", KF);
	waitKey(1);
	
	imageStruct I;

 	MallImage(&I, KF.cols, KF.rows);
 	for (int i = 0; i < KF.rows; i++){  // Across image
 		for (int j = 0; j < KF.cols; j++){ // Down image
 			I.data[i*KF.cols + j] = KF.at<uchar>(i,j);
 		}
 	}

	int miter = 5,  mprec = 2;    
	int posx = 0, posy = 0;
	int sizx = 1280, sizy = 960;
	trackStruct T;
	if (MallTrack (&T, &I, posx, posy, sizx, sizy, miter, mprec))
		return (1);
	else
		printf ("ESM Tracking structure ready\n");

 	for( ;; ){
 		Mat M;
 		cap >> M;
 		cvtColor(M, M, CV_BGR2GRAY);
 		
 		imshow("view", M);
 		waitKey(1);
 		
 		imageStruct I;
 	
	 	MallImage(&I, M.cols, M.rows);
	 	for (int i = 0; i < M.rows; i++){  // Across image
	 		for (int j = 0; j < M.cols; j++){ // Down image
	 			I.data[i*M.cols + j] = M.at<uchar>(i,j);
	 		}
	 	}
	 	
		if (MakeTrack (&T, &I))
			return (1);
		
		
		printf ("%.3f\t%.3f\t%.3f\n", T.homog[0], T.homog[1], T.homog[2]);
		printf ("%.3f\t%.3f\t%.3f\n", T.homog[3], T.homog[4], T.homog[5]);
		printf ("%.3f\t%.3f\t%.3f\n\n", T.homog[6], T.homog[7], T.homog[8]);
	float res = GetZNCC(&T);
	printf("ZNCC is %f\n", res);
	 		
 	}
 	
}
