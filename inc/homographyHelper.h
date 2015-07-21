// Helper function for some heavily used functionality in this code,
// namely finding matching keypoints and finding a homography.

#ifndef HOMOGRAPHY_HELPER_H
#define HOMOGRAPHY_HELPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <vector>
#include <string>

bool matchFeaturesInImages(std::vector<cv::KeyPoint> raw_kps_keyframe, cv::Mat descriptors_keyframe, cv::Mat image,
	int minHessian, cv::Mat &descriptors_moving, std::vector<cv::Point2d> &matched_kps_moved, 
	std::vector<cv::Point2d> &matched_kps_keyframe );
	
bool matchImage(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2d> &matched_kps_moved, 
	std::vector<cv::Point2d> &matched_kps_keyframe);
	
void rotate(cv::Mat& src, double angle);

/************************************************/
	
bool matchFeaturesInImages(std::vector<cv::KeyPoint> raw_kps_keyframe, cv::Mat descriptors_keyframe, cv::Mat image,
	int minHessian, cv::Mat &descriptors_moving, std::vector<cv::Point2d> &matched_kps_moved, 
	std::vector<cv::Point2d> &matched_kps_keyframe ){

	cv::FastFeatureDetector detector( minHessian ); 
	 // Sift seems much less crowded than Surf.
	cv::SurfDescriptorExtractor extractor;  		

	std::vector<cv::KeyPoint> raw_kps_moving;
//	cv::Mat descriptors_moving;
	
	// Detect the feature points and their descriptors in the moving image. 
	detector.detect(image, raw_kps_moving);
	extractor.compute(image, raw_kps_moving, descriptors_moving);		

	/********************************************
	Match the keypoints between the two images so a homography can be made. 
	********************************************/
	cv::BFMatcher matcher;
	std::vector< std::vector< cv::DMatch > > doubleMatches;
	std::vector< cv::DMatch > matches;
		
	/** Option 1: Straight feature matcher. ***/ 
	//matcher.match( descriptors_moving, descriptors_keyframe, matches ); 

	/** Option 2: knn matcher, which matches the best two points and then
	 finds the best of the two. It may be overkill.    **/
	
	matcher.knnMatch( descriptors_moving, descriptors_keyframe, doubleMatches, 2 ); 

	// If the matches are within a certain distance (descriptor space),
	// then call them a good match.

	double feature_distance = 2.0;

	for (unsigned int i = 0; i < doubleMatches.size(); i++) {
		if (doubleMatches[i][0].distance < feature_distance * 
				doubleMatches[i][1].distance){
			matches.push_back(doubleMatches[i][0]);
		}
	}	
	
	/** End of Option 2  **/

	double max_dist = 0; double min_dist = 100;

	// Regardless of option picked, calculate max and min distances (descriptor space) between keypoints
	for( int i = 0; i < descriptors_moving.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	// Store only "good" matches (i.e. whose distance is less than minDistMult*min_dist )
	std::vector< cv::DMatch > good_matches;

	double minDistMult = 2.0;

	for( int i = 0; i < descriptors_moving.rows; i++ ){ 
		if( matches[i].distance < minDistMult * min_dist ) { 
			good_matches.push_back( matches[i] );
		}
	}
	
	/********************************************
	End of keypoint matching
	********************************************/

	/********************************************
	Copy the keypoints from the matches into their own vectors. 
	This is necessary because I don't want to take out any detected
	keypoints from the keyframe match, since I use that repeatedly.
	*********************************************/
	// Vectors that contain the matched keypoints. 
	//std::vector<cv::Point2d> matched_kps_moved;
	//std::vector<cv::Point2d> matched_kps_keyframe;

	for( int i = 0; i < (int) good_matches.size(); i++ )
	{
	  matched_kps_moved.push_back( raw_kps_moving[ good_matches[i].queryIdx ].pt );  // Left frame
	  matched_kps_keyframe.push_back( raw_kps_keyframe[ good_matches[i].trainIdx ].pt );
	}
	
	// OK, so now we have two vectors of matched keypoints. 
	if (! (matched_kps_moved.size() < 4 || matched_kps_keyframe.size() < 4) ){
		std::vector<uchar> status; 
	
		double fMatP1 = 2.0;
		double fMatP2 = 0.99;
		
	// Use RANSAC and the fundamental matrix to take out points that don't fit geometrically
		findFundamentalMat(matched_kps_moved, matched_kps_keyframe,
			CV_FM_RANSAC, fMatP1, fMatP2, status);

		// Erase any points from the matched vectors that don't fit the fundamental matrix
		// with RANSAC.
		for (int i = matched_kps_moved.size() - 1; i >= 0; i--){
			if (status[i] == 0){
				matched_kps_moved.erase(matched_kps_moved.begin() + i);
				matched_kps_keyframe.erase(matched_kps_keyframe.begin() + i);
			}
		}
		return true;
	}else{
		return false;
	}
	
	return false;
	
	// Return: matched_kps_moved, descriptors_moving

}

bool matchImage(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2d> &matched_kps_moved, 
	std::vector<cv::Point2d> &matched_kps_keyframe, int minHessian){
	
	cv::FastFeatureDetector detector( minHessian ); 
	 // Sift seems much less crowded than Surf.
	cv::SurfDescriptorExtractor extractor;  		

	std::vector<cv::KeyPoint> raw_kps_keyframe;
	cv::Mat descriptors_keyframe;
	detector.detect(image1, raw_kps_keyframe);  
	extractor.compute(image1, raw_kps_keyframe, descriptors_keyframe); 
	
	cv::Mat descriptors_moving;
	
	if( matchFeaturesInImages(raw_kps_keyframe, descriptors_keyframe, image2, minHessian, descriptors_moving, 
		matched_kps_moved, matched_kps_keyframe ) ){
		return true;	
	}else{
		return false;
	}
	
}

void rotate(cv::Mat& src, double angle)
{		// Rotation angle is in degrees.
//	int len = std::max(src.cols, src.rows);
	cv::Point2f pt(src.cols/2., src.rows/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

	cv::warpAffine(src, src, r, cv::Size(src.cols, src.rows));
}


#endif
