#include <iostream>  
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/objdetect/objdetect.hpp"
#include <string>
#include <fstream>
#include "antiSpofModel.h"
#include "featureExtractor.h"
#include "faceDetector.h"
#include <opencv2/ml.hpp>
#include "linear.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define Random(x) (rand() % x)

int featureDim = 384;
int numPcaBase = 300;
int numGmmClusters = 128;


void load_svm(svmModel*& svm, string filename)
{
	ifstream fin(filename);
	if (!fin)
	{
		std::cout << "Unable to open file: svm.dat " << endl;
		exit(1);
	}

	int nr_feature;
	int n;
	int nr_class;
	double bias;
	parameter & param = svm->param;
	svm->label = NULL;
	string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "solver_type")
		{
			fin >> cmd;
			for (int i = 0; solver_type_table[i]; i++)
			{
				if (solver_type_table[i] == cmd)
				{
					param.solver_type = i;
					break;
				}
			}
		}
		else if (cmd == "nr_class")
		{
			fin >> nr_class;
			svm->nr_class = nr_class;
		}
		else if (cmd == "nr_feature")
		{
			fin >> nr_feature;
			svm->nr_feature = nr_feature;
		}
		else if (cmd == "bias")
		{
			fin >> bias;
			svm->bias = bias;

		}
		else if (cmd == "w")
		{
			nr_feature = svm->nr_feature;
			if (svm->bias >= 0)
				n = nr_feature + 1;
			else
				n = nr_feature;
			int w_size = n;
			int nr_w;
			if (nr_class == 2 && param.solver_type != MCSVM_CS)
				nr_w = 1;
			else
				nr_w = nr_class;
			svm->w = Malloc(double, w_size*nr_w);
			for (int i = 0; i < w_size; i++)
			{
				for (int j = 0; j < nr_w; j++)
				{
					fin >> svm->w[i*nr_w + j];
				}
			}
			break;
		}
		else if (cmd == "label")
		{
			int nr_class = svm->nr_class;
			svm->label = Malloc(int, nr_class);
			for (int i = 0; i < nr_class; i++)
			{
				fin >> svm->label[i];
			}
		
		}
		else
		{
			cout << "unknown text in model file: " << cmd << endl;
			free(svm->label);
			free(svm->w);
			return;
		}
	}
	fin.close();
}


void load_pca(cv::PCA& pca, string filename)
{
	ifstream fin(filename);
	if (!fin)
	{
		std::cout << "Unable to open file: pca.dat " << endl;
		exit(1);
	}
	string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "eigenvectors")
		{
			cv::Mat eigenvectors(numPcaBase, featureDim, CV_64F);
			for (int r = 0; r < eigenvectors.rows; r++)
			{
				for (int c = 0; c < eigenvectors.cols; c++)
				{
					fin >> eigenvectors.at<double>(r, c);
				}
			}
			pca.eigenvectors = eigenvectors;
		}
		else if (cmd == "eigenvalues")
		{
			cv::Mat eigenvalues(numPcaBase, 1, CV_64F);
			for (int r = 0; r < eigenvalues.rows; r++)
			{
				for (int c = 0; c < eigenvalues.cols; c++)
				{
					fin >> eigenvalues.at<double>(r, c);
				}
			}
			pca.eigenvalues = eigenvalues;

		}
		else if (cmd == "mean")
		{
			cv::Mat mean(featureDim, 1, CV_64F);
			for (int r = 0; r< mean.rows; r++)
			{
				for (int c = 0; c < mean.cols; c++)
				{
					fin >> mean.at<double>(r, c);
				}
			}
			pca.mean = mean;
			break;

		}
		else
		{
			cout << "unknown text in pca.dat: " << cmd << endl;
			exit(1);
		}

	}
	fin.close();

}

void load_gmm(VlGMM*& gmm, string filename)
{

	ifstream fin(filename);
	if (!fin)
	{
		std::cout << "Unable to open file: gmm.dat " << endl;
		exit(1);
	}
	string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "gmmPriors")
		{
			gmm = vl_gmm_new(VL_TYPE_DOUBLE, numPcaBase, numGmmClusters);
			vl_gmm_set_initialization(gmm, VlGMMCustom);

			double * gmmPriors = (double *)vl_calloc(numGmmClusters, sizeof(double));
			for (int i = 0; i < numGmmClusters; i++)
			{
				fin >> gmmPriors[i];
			}
			vl_gmm_set_priors(gmm, gmmPriors);

		}
		else if (cmd == "gmmMeans")
		{
			double * gmmMeans = (double *)vl_calloc(numGmmClusters*numPcaBase, sizeof(double));
			for (int i = 0; i < numGmmClusters*numPcaBase; i++)
			{
				fin >> gmmMeans[i];
			}
			vl_gmm_set_means(gmm, gmmMeans);

		}
		else if (cmd == "gmmCovariances")
		{
			double * gmmCovariances = (double *)vl_calloc(numGmmClusters*numPcaBase, sizeof(double));
			for (int i = 0; i < numGmmClusters*numPcaBase; i++)
			{
				fin >> gmmCovariances[i];
			}
			vl_gmm_set_covariances(gmm, gmmCovariances);
			break;

		}
		else
		{
			cout << "unknown text in model file: " << cmd << endl;
			vl_gmm_delete(gmm);
			exit(1);
		}
	}
	fin.close();
}



void gmm_coding(cv::Mat&in, cv::Mat&fv, VlGMM*& gmm)
{
	vl_size dimension = numPcaBase;
	vl_size numClusters = numGmmClusters;
	double * enc = (double *)vl_malloc(sizeof(double)* 2 * dimension*numClusters);

	int localFeatureNum = in.rows;
	int localFeatureDim = in.cols;

	double * data = (double *)vl_malloc(sizeof(double)*localFeatureNum*localFeatureDim);
	for (int i = 0; i < in.cols; i++)
	{
		for (int j = 0; j<in.rows; j++)
		{
			data[i*in.rows + j] = in.at<double>(j, i);
		}
	}
	vl_fisher_encode(enc, VL_TYPE_DOUBLE,
		vl_gmm_get_means(gmm), dimension, numClusters,
		vl_gmm_get_covariances(gmm),
		vl_gmm_get_priors(gmm),
		data, in.rows,
		VL_FISHER_FLAG_NORMALIZED | VL_FISHER_FLAG_SQUARE_ROOT | VL_FISHER_FLAG_IMPROVED);
	fv = cv::Mat(2 * dimension*numClusters, 1, CV_64F);
	for (int i = 0; i<2 * dimension*numClusters; i++)
	{
		fv.at<double>(i) = enc[i];
	}
	vl_free(enc);
	vl_free(data);

}

void code_into_sparse(struct feature_node * &x, cv::Mat &code)
{
	cv::Mat enc = code.t();

	int cnt = 0;
	for (int i = 0; i < enc.rows; ++i)
	{
		if (fabs(enc.at<double>(i)) > 1e-6)
		{
			++cnt;
		}
	}

	x = Malloc(struct feature_node, cnt + 1);

	cnt = 0;
	for (int i = 0; i< enc.rows; ++i)
	{
		if (fabs(enc.at<double>(i)) > 1e-6)
		{
			x[cnt].value = enc.at<double>(i);
			x[cnt].index = i + 1;
			++cnt;
		}
	}

	x[cnt].index = -1;
}

int predictLabel(struct model *& svm, cv::Mat& code)
{
	assert(!code.empty());
	code = code.t();
	struct feature_node *x;
	code_into_sparse(x, code);
	double label = predict(svm, x);
	free(x);
	return int(label);
}

int main()
{
	string svm_model = "./surf11_pca300_balance/svm_7.dat";
	string pca_model = "./surf11_pca300_balance/pca.dat";
	string gmm_model = "./surf11_pca300_balance/gmm.dat";

	svmModel* svm = Malloc(svmModel, 1);
	cv::PCA pca;
	VlGMM* gmm;
	

	load_svm(svm, svm_model);
	load_pca(pca, pca_model);
	load_gmm(gmm, gmm_model);



	std::string  face_cascade_name = "haarcascade_frontalface_alt.xml";
	faceDetector face_cascade(face_cascade_name);
	featureExtractor extractor;

	VideoWriter writer("VideoTest_yushe.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));

	VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	double rate = 25.0;
	Size videoSize(640, 480);
	cv::Mat frame;
	vector<Rect> faces;
	int label;
	bool isFirstFrame = true;

	if (capture.isOpened())
	{
		while (true)
		{
			capture >> frame;
			if (isFirstFrame)
			{
				isFirstFrame = false;
				continue;
			}
			if (frame.empty())
				break;

			faces = face_cascade.detect(frame);
			if (faces.size() > 0)
			{
				for (size_t i = 0; i < faces.size(); i++)
				{
					int xo = faces[i].x + faces[i].width / 2;
					int yo = faces[i].y + faces[i].height / 2;
					int L = int(max(faces[i].width, faces[i].height)*1.2);
					cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));
					cv::Mat face = frame(rect);
					cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
					cv::Mat  Feature = extractor.findSurfDescriptor(face);
					cv::Mat pcaCode = pca.project(Feature);
					cv::Mat gmmCode;
					gmm_coding(pcaCode, gmmCode, gmm);
					label = predictLabel(svm, gmmCode);
					cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
					if (label == 1)
					{
						cv::putText(frame, "Geninue", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
					}
					else
					{
						cv::putText(frame, "Fake", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
					}


				}
			}
			cv::imshow("Video", frame);
			writer << frame;
			if (waitKey(1) == 27)
			{
				break;
			}
		}
	}
	else
	{
		vl_gmm_delete(gmm);
		free_and_destroy_model(&svm);
		cout << "--��!�� No capture id : 0 " << endl;
		system("pause");
		exit(1);
	}
	vl_gmm_delete(gmm);
	free_and_destroy_model(&svm);
}