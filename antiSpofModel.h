#ifndef _ANTI_SPOF_MODEL_
#define _ANTI_SPOF_MODEL_
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "linear.h"
#include "tron.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <vl/host.h>
#include <vl/fisher.h>
#include <vl/gmm.h>
using namespace std;
class antiSpofModel
{
public:
	antiSpofModel(){};
	~antiSpofModel()
	{
		vl_gmm_delete(gmm);
	};
	void /*antiSpofModel::*/load(std::string filename)
	{
		ifstream fin(filename);
		if(!fin)
	   {
		    std::cout << "Unable to open file: " << filename <<endl;
		    exit(1);
	    }

		int nr_feature;
		int n;
		int nr_class;
		double bias;
		parameter & param = svm.param;
		svm.label = NULL;	
		string cmd;
		while(1)
		{
			fin >> cmd;
			if(cmd == "featureDim")
			{
				fin >> featureDim;
			}
			else if(cmd == "numPcaBase")
			{
				fin >> numPcaBase;
			}
			else if(cmd == "gmmDimension")
			{
				fin >> gmmDimension;
			}
			else if(cmd == "numGmmClusters")
			{
				fin >> numGmmClusters;
			}
			else if(cmd == "gmmFlag")
			{
				fin >> gmmFlag;
			}
			else if(cmd == "VL_DATA_TYPE")
			{
				fin >> VL_DATA_TYPE;
			}
			else if(cmd == "solver_type")
			{
				fin >> cmd;
				for(int i = 0; solver_type_table[i]; i++)
				{
					if(solver_type_table[i] == cmd)
					{
						param.solver_type = i;
						break;
					}
				}
			}
			else if(cmd == "nr_class")
			{
				fin >> nr_class;
				svm.nr_class = nr_class;
			}
			else if(cmd == "nr_feature")
			{
				fin >> nr_feature;
				svm.nr_feature = nr_feature;
			}
			else if(cmd == "bias")
			{
				fin >> bias;
				svm.bias = bias;

			}
			else if(cmd == "w")
			{
				nr_feature = svm.nr_feature;
				if(svm.bias >= 0)
					n = nr_feature+1;
				else
					n = nr_feature;
				int w_size = n;
				int nr_w;
				if(nr_class == 2 && param.solver_type != MCSVM_CS)
					nr_w = 1;
				else
					nr_w = nr_class;
				svm.w = Malloc(double,w_size*nr_w);
				for(int i = 0; i< w_size; i++)
				{
					for(int j = 0;j < nr_w; j++)
					{
						fin >> svm.w[i*nr_w+j];
					}
				}
			}
			else if(cmd == "label")
			{
				int nr_class = svm.nr_class;
				svm.label = Malloc(int,nr_class);
				for(int i = 0; i < nr_class; i++)
				{
					fin >> svm.label[i];
				}
			}
			else if(cmd == "eigenvectors")
			{
				cv::Mat eigenvectors(numPcaBase,featureDim,CV_64F);
				for(int r = 0; r < eigenvectors.rows; r++)
				{
					for(int c = 0;c < eigenvectors.cols; c++)
					{
						fin >> eigenvectors.at<double>(r,c);
					}
				}
				pca.eigenvectors = eigenvectors;
			}
			else if(cmd == "eigenvalues")
			{
				cv::Mat eigenvalues(numPcaBase,1,CV_64F);
				for(int r = 0; r < eigenvalues.rows;r++)
				{
					for(int c = 0; c < eigenvalues.cols;c++)
					{
						fin >> eigenvalues.at<double>(r,c);
					}
				}
				pca.eigenvalues = eigenvalues;
			}
			else if(cmd == "mean")
			{
				cv::Mat mean(featureDim,1,CV_64F);
				for(int r = 0 ; r< mean.rows; r++)
				{
					for(int c = 0; c < mean.cols; c++)
					{
						fin >> mean.at<double>(r,c);
					}
				}
				pca.mean = mean;
			}
			else if(cmd == "gmmPriors")
			{
				gmm = vl_gmm_new(VL_DATA_TYPE,gmmDimension,numGmmClusters);
				vl_gmm_set_initialization(gmm,VlGMMCustom);

				double * gmmPriors  = (double * )vl_calloc(numGmmClusters,sizeof(double));
				for(int i = 0;i < numGmmClusters; i++)
				{
					fin >> gmmPriors[i];
				}
				vl_gmm_set_priors(gmm,gmmPriors);
			}
			else if(cmd == "gmmMeans")
			{
				double * gmmMeans = (double * )vl_calloc(numGmmClusters*gmmDimension,sizeof(double));
				for(int i = 0;i < numGmmClusters*gmmDimension; i++)
				{
					fin >> gmmMeans[i];
				}
				vl_gmm_set_means(gmm,gmmMeans);
			}
			else if(cmd == "gmmCovariances")
			{
				double * gmmCovariances = (double * )vl_calloc(numGmmClusters*gmmDimension,sizeof(double));
				for(int i = 0; i < numGmmClusters*gmmDimension; i++)
				{
					fin >> gmmCovariances[i];
				}
				vl_gmm_set_covariances(gmm,gmmCovariances);
				break;
			}
			else
			{
				cout << "unknown text in model file: " << cmd <<endl;
				free(svm.label);
				vl_gmm_delete(gmm);
				return ;
			}
		}


		fin.close();
	}
	void /*antiSpofModel::*/save(std::string filename)
	{
		ofstream fout(filename);
		if(!fout)
		{
			 std::cout << "Unable to open file: " << filename <<endl;
		     exit(1);
		}
		//save general params
		fout << "featureDim " << featureDim << endl;
		fout << "numPcaBase " << numPcaBase << endl;
		fout << "gmmDimension " << gmmDimension << endl;
		fout << "numGmmClusters " << numGmmClusters << endl;
		fout << "gmmFlag " << gmmFlag << endl;
		fout << "VL_DATA_TYPE " << VL_DATA_TYPE << endl;

		//save svm
		int i;
		int nr_feature = svm.nr_feature;
		int n ;
		const  parameter& param = svm.param;
		if(svm.bias>=0)
			n = nr_feature+1;
		else
			n = nr_feature;
		int w_size = n;
		int nr_w;
		if(svm.nr_class == 2 && svm.param.solver_type != MCSVM_CS)
			nr_w = 1;
		else 
			nr_w =svm.nr_class;
		fout << "solver_type " << solver_type_table[param.solver_type] <<endl;
		fout << "nr_class " << svm.nr_class << endl;
		if(svm.label)
		{
			fout << "label";
			for(i = 0;i<svm.nr_class;i++)
			{
				fout << " " << svm.label[i];
			}
			fout << endl;
		}
		fout << "nr_feature " << nr_feature << endl;
		fout << "bias " << fixed << setprecision(16) << svm.bias <<endl;
		fout << "w" <<endl;
		for(i = 0 ;i < w_size; i++)
		{
			for(int j = 0;j < nr_w; j++)
			{
				fout << fixed << setprecision(16) << svm.w[i*nr_w+j] << " ";
			}
		}
		fout << endl;
	// save pca
		cv::Mat eigenvectors = pca.eigenvectors;
		fout << "eigenvectors" <<endl;
		for(int r = 0 ;r < eigenvectors.rows; r++)
		{
			for(int c = 0; c < eigenvectors.cols; c++)
			{
				fout << fixed << setprecision(16) << eigenvectors.at<double>(r,c) << " ";
			}
		}
		fout << endl;
		cv::Mat eigenvalues = pca.eigenvalues;
		fout << "eigenvalues" << endl;
		for(int r = 0; r < eigenvalues.rows; r++)
		{
			for(int c = 0; c < eigenvalues.cols; c++)
			{
			    fout << fixed << setprecision(16) << eigenvalues.at<double>(r,c) << " ";
			}
		}
		fout << endl;
		cv::Mat mean = pca.mean;
		fout << "mean" << endl;
		for(int r = 0; r < mean.rows; r++)
		{
			for(int c = 0; c < mean.cols; c++)
			{
				fout << fixed << setprecision(16) << mean.at<double>(r,c) << " ";
			}
		}
		fout << endl;

		//save gmm
		double *gmmPriors = (double * )vl_gmm_get_priors(gmm);
		fout << "gmmPriors" <<endl;
		for(i = 0; i < numGmmClusters; i++)
		{
			fout << fixed << setprecision(16) << gmmPriors[i] << " ";
		}
		fout << endl;
		double *gmmMeans = (double *)vl_gmm_get_means(gmm);
		fout << "gmmMeans" << endl;
		for(i = 0; i < numGmmClusters*gmmDimension; i++)
		{
			fout << fixed << setprecision(16) << gmmMeans[i] << " ";
		}
		fout << endl;
		double *gmmCovariances = (double *)vl_gmm_get_covariances(gmm);
		fout << "gmmCovariances" << endl;
		for(i = 0;i < numGmmClusters*gmmDimension; i++)
		{
			fout << fixed << setprecision(16) << gmmCovariances[i] << " ";
		}
		fout << endl;
		fout.close();
	}
	//tools
	svmModel& /*antiSpofModel::*/get_svmModel()
	{
		return svm;
	}
	cv::PCA& /*antiSpofModel::*/get_pcaModel()
	{
		return pca;
	}
	VlGMM*& /*antiSpofModel::*/get_gmmModel()
	{
		return gmm;
	}

private:
	svmModel svm;
	cv::PCA pca;
	VlGMM *gmm;

public:
	vl_size numGmmClusters;
	vl_size gmmDimension;
	int gmmFlag;
	int VL_DATA_TYPE;
	int featureDim ;
	int numPcaBase;
};

#endif