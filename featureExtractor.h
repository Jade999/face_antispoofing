#ifndef _FEATURE_EXTRACTOR_H
#define _FEATURE_EXTRACTOR_H
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include"opencv2/xfeatures2d.hpp"  
#include "opencv2/imgcodecs.hpp"
#include<opencv2/opencv.hpp>  
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

cv::Mat mergeRows(cv::Mat A, cv::Mat B)
{
			assert(A.cols == B.cols&&A.type() == B.type());
		    int totalRows = A.rows + B.rows;
		    cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
		    cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
            A.copyTo(submat);
            submat = mergedDescriptors.rowRange(A.rows, totalRows);
            B.copyTo(submat);
            return mergedDescriptors;
}

class featureExtractor
{
public:
	featureExtractor()
	{
		surf = SURF::create();
		surf->setExtended(false);
		surf->setUpright(true);
	
		pointStep = 2;
		imgSize = 64;
		featureDim = 384;

		densePoints.clear();
		for (int y = 0; y < imgSize; y += pointStep)
		{
			for (int x = 0; x < imgSize; x += pointStep)
			{
			    densePoints.push_back(KeyPoint(float(x),float(y),float(11)));
			}
		}
		numDensePoint = densePoints.size();
	}
	cv::Mat featureExtractor::findSurfDescriptor(cv::Mat& img)
	{
		cv::Mat hsv, ycrcb;
		cvtColor(img, hsv, CV_BGR2HSV);
		cvtColor(img,ycrcb,CV_BGR2YCrCb);

		vector<cv::Mat> channels ;
		cv::Mat code;
		cv::split(hsv,channels);

		cv::Mat Feature;
		for(size_t i = 0;i < channels.size(); ++i )
		{
			surf->compute(channels[i],densePoints,code);
			if(Feature.empty())
			{
				Feature = code.t()  ;
			}
			else
			{
				Feature = mergeRows(Feature,code.t());
			}
		
		}

		cv::split(ycrcb,channels);
		for(size_t i = 0;i < 3;++i )
		{
			surf->compute(channels[i],densePoints,code);
	        if(Feature.empty())
			{
				Feature = code.t();
			}
			else
			{
				Feature = mergeRows(Feature,code.t());
			}
		}
		
		Feature.convertTo(Feature,CV_64F);
		return Feature;
	}
	//std::vector<cv::KeyPoint> featureExtractor::findDenseKeyPoint(cv::Mat& img)
	//{
	//	cv::Mat img_gray;
	//	cv::DenseFeatureDetector dense(1.0f,1,0.1f,2);
	//	cv::vector<cv::KeyPoint> key_points;  
	//	cv::cvtColor( img, img_gray, CV_BGR2GRAY ); 
	//	dense.detect(img_gray,key_points,cv::Mat());
	//    return key_points;
	//}

	int featureExtractor::get_featureDim()
	{
		return featureDim;
	}
	int featureExtractor::get_numDensePoint()
	{
		return numDensePoint;
	}

private:
	Ptr<SURF> surf;
	std::vector<cv::KeyPoint> densePoints;
	int pointStep;
	int imgSize;

	int featureDim;
	int numDensePoint;
};

#endif