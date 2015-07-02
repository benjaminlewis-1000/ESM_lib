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
	
 	Mat M = imread( "test.jpg" );
 	
 	cvtColor(M, M, CV_BGR2GRAY);
 	
 	imshow("Test",M);
 	waitKey(0);
 	
 	imageStruct I;
 	
  	char filename[50];
 	sprintf (filename, "./res/patr.pgm");
 	
 	MallImage(&I, M.cols, M.rows);
 	for (int i = 0; i < M.rows; i++){  // Across image
 		for (int j = 0; j < M.cols; j++){ // Down image
 			I.data[i*M.cols + j] = M.at<uchar>(i,j);
 		}
 	}
 	
 	
 	
 	SavePgm(filename, &I);
}
