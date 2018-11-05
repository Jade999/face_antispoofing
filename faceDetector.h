#ifndef _FACE_DETECTOR_H
#define _FACE_DETECTOR_H

#include <string>
#include <iostream>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

class faceDetector
{
public:
	 faceDetector(std::string filename)
	{
		model_path = filename;
	   if( !face_cascade.load( model_path)  )
		{ 
			printf("--(!)Error loading\n");
			system("PAUSE");
			exit(1);
		};
	}

	std::vector<cv::Rect>& detect(cv::Mat &img)
	{
		cv::Mat img_gray;
		cvtColor(img,img_gray,CV_BGR2GRAY);
	    face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
		return faces;
	}
private:
	std::string model_path;
	cv::CascadeClassifier face_cascade;
	std::vector<cv::Rect> faces;

};

#endif