#include <opencv2/opencv.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include<vector>
#include<mat.h>
#include<io.h>
using namespace std;
using namespace cv;

#define FEATURE_NUM 100
#define DEBUG

/**Usage*/
static void help()
{
	printf("\nThis is picture recognition demo.\n"
		"The usage: svm [ -train <path to pictures for training classifier>] each class in a seperate dir\\\n"
		"  [-anno <path to annotations that contains ROI of pictures for training samples>] each class in a seperate dir\\\n"
		"  [-predict <path to pictures for training classifier>] each class in a seperate dir\\\n"
		"  [-predict_anno <path to annotations that contains ROI of pictures for predicting samples>] each class in a seperate dir\\\n"
		"  [-save <output XML file for the classifier>] \\\n"
		"  [-load <XML file with the pre-trained classifier>] \\\n"
		"  [-boost|-svm] # choose which classifier to use, default classifier is svm \n");
}

/**
*Read file names and numbers in given path
*
*@param [in]  path  file path
*@param [out] files file names
*@param [out] info  file numbers in each dir
*/
static void getFiles(string path, vector<string>& files, vector<int> &info){
	if (sizeof(int*) == 8){
#define x64
	}
	else{
#define x86
	}
	int numFiles = 0;
	//file handler  
	long long   hFile = 0;
	string p;
#ifdef x64
	//file info
	struct __finddata64_t fileinfo;
	if ((hFile = _findfirsti64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
#else
	//file info
	struct __finddata_t fileinfo;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
#endif
		do
		{
			//recurse if it's dir
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files, info);
			}
			else
			{
				numFiles++;
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
#ifdef x64
		} while (_findnexti64(hFile, &fileinfo) == 0);
#else
		} while (_findnext(hFile, &fileinfo) == 0);
#endif
		_findclose(hFile);
		if (numFiles != 0)
			info.push_back(numFiles);
	}
}

/**mormalize data by column*/
static void normalizecol(Mat &src){
	Mat temp;
	Mat avg; //column avg
	Mat std; //column std
	Mat avg_tmp; //src - avg

	//calculate avg
	reduce(src, temp, 0, CV_REDUCE_SUM);
	avg = temp / src.rows;
	repeat(avg, src.rows, 1, avg);

	//calculate std
	subtract(src, avg, avg_tmp);//src - avg
	pow(avg_tmp, 2, temp);
	reduce(temp, temp, 0, CV_REDUCE_SUM); // calculate sum of each col 
	sqrt(temp, temp);
	std = temp / (src.rows - 1);
	repeat(std, src.rows, 1, std);

	divide(avg_tmp, std, src);
}

/**
*Detect picture features, set roi and smooth pictures with CV_GAUSSIAN 3*3,
*then detect feature with sift algorithms and normalize data by column.
***************************************************************************
*To use all the pictures for one purpose(training or predicting):         *
*detectFeatures(pic_path,annotation_path,responses,data,Mat(),Mat(),1);   *
***************************************************************************
*
*@param [in]  pic_path      path of pictures
*@param [out] response      class of pictures for training samples
*@param [out] data          features of pictures for training samples
*@param [out] pre_responses class of pictures for predicting samples
*@param [out] pre_data      features of pictures for predicting samples
*@param [in]  factor        percentage of training samples
*/
static int detectFeatures(const char* pic_path,
	const char* annotation_path,
	Mat &responses,
	Mat &data,
	Mat &pre_responses,
	Mat &pre_data,
	double factor = 0.8){
	SiftFeatureDetector  siftdtc(100);
	SurfDescriptorExtractor extractor;
	Mat descriptor;

	int feature_num;

	vector<string> files;
	vector<int> info;
	vector<string> anno_files;
	vector<int> anno_info;	
	getFiles(pic_path, files, info);
	getFiles(annotation_path, anno_files, anno_info);

	for (int i = 0; i < info.size(); i++){
		printf("num class[%d]: %d\n", i, info[i]);
	}
	for (int i = 0; i < files.size(); i++)
		cout << files[i] << endl;

	printf("detecting training picture features (may take a few minutes)...\n");
	for (int i = 0, j = 0, t = 0; i < files.size(); i++, j++){
		Mat m_img = imread(files[i]);

		//read roi from files
		MATFile *pmatFile = NULL;
		mxArray *pmxArray = NULL;
		double* initA = NULL;
		pmatFile = matOpen(anno_files[i].c_str(), "r");
		pmxArray = matGetVariable(pmatFile, "box_coord");
		initA = (double*)mxGetData(pmxArray);

		IplImage* iplImage = new IplImage(m_img);
		cvSetImageROI(iplImage, cvRect(initA[2], initA[0], initA[3] - initA[2], initA[1] - initA[0]));
		cvSmooth(iplImage, iplImage, CV_GAUSSIAN, 3, 3);
		Mat img = Mat(iplImage);
		vector<KeyPoint>kp1;
		siftdtc.detect(img, kp1);
		extractor.compute(img, kp1, descriptor);

		if (descriptor.rows != FEATURE_NUM){
			//skip pictures that do not have exactly FEATURE_NUM features
			matClose(pmatFile);
			continue;
		}
		//write responses
		if (j >= info[t]){
			j = 0;
			t++;
		}

#ifdef DEBUG
		printf("processing: %s class:%d\n", files[i].c_str(), t);
		printf("feature size:[%d,%d]\n", descriptor.rows, descriptor.cols);
#endif

		feature_num = descriptor.rows*descriptor.cols;
		Mat des = descriptor.reshape(0, feature_num);
		if (j < (int)(info[t] * factor)){
			responses.push_back(t);
			data.push_back(des);
		}
		else{
			pre_responses.push_back(t);
			pre_data.push_back(des);
		}

		matClose(pmatFile);
	}/*end of for (int i = 0, j = 0, t = 0, m = 0; i < files.size(); i++,j++)*/

	data = data.reshape(0, feature_num);
	data = data.t();
	normalizecol(data);
	if ((int)factor != 1){
		pre_data = pre_data.reshape(0, feature_num);
		pre_data = pre_data.t();
		normalizecol(pre_data);
	}

#ifdef DEBUG
	FileStorage hhh("features.xml", FileStorage::WRITE);
	hhh << "responses" << responses;
	hhh << "data" << data;
	if ((int)factor != 1){
		hhh << "pre_responses" << pre_responses;
		hhh << "pre_data" << pre_data;
	}
	for (int i = 0; i < info.size(); i++){
		printf("num class[%d]: %d\n", i, info[i]);
	}
		printf("\n\ntrain num: %d\n", data.rows);
		printf("predict num: %d\n\n", pre_data.rows);
#endif

	return 0;
}

/**
*Same as previous function, except that all the pictures are used for only one purpose(training or predicting): 
*/
static int detectFeatures(const char* pic_path,const char* annotation_path, Mat responses, Mat data){
	return detectFeatures(pic_path, annotation_path, responses, data, Mat(), Mat(), 1);
}

/**svm classifier
*
*@param [in]  train_pic_path         path of pictures for training
*@param [in]  annotation_path        path of annotation for training
*@param [in]  classfication_pic_path path of pictures for predicting
*@param [in]  predict_anno_path      path of annotation for predicting
*@param [in]  finename_to_load       path of classifier to load
*@param [out] filename_to_save       path of classifier to save
*/
static int svm_classifier(const char* train_pic_path,
	const char* annotation_path,
	const char* classfication_pic_path,
	const char* predict_anno_path,
	const char* filename_to_save,
	const char* filename_to_load){

	CvSVM svm;
	CvSVMParams param;
	param.kernel_type = CvSVM::LINEAR;
	param.svm_type = CvSVM::C_SVC;
	param.C = 1;

	if (filename_to_load){
		// load classifier from the specified file
		svm.load(filename_to_load);
		if (svm.get_var_count() == 0){
			printf("Could not read the classifier %s\n", filename_to_load);
			return -1;
		}
		printf("The classifier %s is loaded.\n", filename_to_load);
	}
	else{
		if (!train_pic_path || !annotation_path){
			printf("Don't have a train pictures path or annotation path and path to load classifier\n");
			help();
			return -1;
		}
		Mat data, responses;
		detectFeatures(train_pic_path, annotation_path, responses, data);
		printf("Training the classifier (may take a few minutes)...\\\n");
		svm.train(data, responses, Mat(), Mat(), param);
		if (filename_to_save)
			svm.save(filename_to_save);
	}

	if (classfication_pic_path && predict_anno_path){
		printf("Classification (may take a few minutes)...\n");

		//get features
		Mat classi_data;
		Mat true_responses;
		detectFeatures(classfication_pic_path, predict_anno_path, true_responses, classi_data);
		Mat result(1, true_responses.cols, CV_32S);

		//predict
		double t = (double)cvGetTickCount();
		svm.predict(classi_data, result);
		t = (double)cvGetTickCount() - t;
		printf("Prediction type: %gms\n", t / (cvGetTickFrequency()*1000.));

		int true_resp = 0;
		for (int i = 0; i < true_responses.rows; i++){
			if ((int)result.at<float>(i) == (int)true_responses.at<float>(i))
				true_resp++;
		}

#ifdef DEBUG
		//write result to file
		FileStorage result_file("result.xml", FileStorage::WRITE);
		result_file << "result" << result;
		result_file << "true_responses" << true_responses;
		result_file << "true_resp" << true_resp;
		result_file << "classi_data" << classi_data;
#endif
		printf("true_resp = %f%%\n", (float)true_resp / true_responses.rows * 100);
	}
	return 0;
}

/**TODO(Robort): To be implement*/
static int boost_classifier(const char* pic_path,
	const char* annotation_path,
	const char* classification_pic_path,
	const char* predict_anno_path,
	const char* filename_to_save,
	const char* filename_to_load){
	SiftFeatureDetector  siftdtc(FEATURE_NUM);
	SurfDescriptorExtractor extractor;
	Mat descriptor;

	int feature_num;
	int num_classes;
	vector<string> files;
	vector<int> info;
	getFiles(pic_path, files, info);

	num_classes = info.size();
	Mat responses(num_classes*files.size(), 1, CV_32S);
	Mat data;

	for (vector<string>::iterator it = files.begin(); it < files.end(); it++){
		Mat img = imread(*it);
		vector<KeyPoint>kp1;
		siftdtc.detect(img, kp1);
		extractor.compute(img, kp1, descriptor);
		feature_num = descriptor.rows*descriptor.cols;
		Mat des = descriptor.reshape(0, feature_num);
		Mat padding(1, 1, CV_32F);
		des.push_back(padding);

		for (int j = 0; j < num_classes; j++){
			des.at<float>(feature_num) = j;
			data.push_back(des);
		}
	}

	for (int t = 0; t < files.size(); t++){
		int num_samples_per_class = files.size() / num_classes;
		for (int j = 0; j < num_classes; j++)
			responses.at<int>(t*num_classes + j) = t / num_samples_per_class;
	}

	CvBoost boost;
	Mat var_type(feature_num + 2, 1, CV_8U, Scalar(CV_VAR_ORDERED));
	data = data.reshape(0, feature_num + 1);

	// 3. train classifier
	printf("Training the classifier (may take a few minutes)...\n");
	boost.train(data, CV_COL_SAMPLE, responses, Mat(), Mat(), var_type, Mat(), CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0));

	return 0;
}

/**program driver*/
static int run(int argc, char * argv[]){
	int method = 1; //default classifier svm
	char* filename_to_save = 0;
	char* filename_to_load = 0;
	char* train_pic_path = 0;
	char* classification_pic_path = 0;
	char* annotation_path = 0;
	char* predict_anno_path = 0;
	int i;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-save") == 0) // flag "-save filename.xml"
		{
			i++;
			filename_to_save = argv[i];
		}
		else if (strcmp(argv[i], "-load") == 0) // flag "-load filename.xml"
		{
			i++;
			filename_to_load = argv[i];
		}
		else if (strcmp(argv[i], "-boost") == 0)
		{
			method = 1;
		}
		else if (strcmp(argv[i], "-anno") == 0)
		{
			i++;
			annotation_path = argv[i];
		}
		else if (strcmp(argv[i], "-svm") == 0)
		{
			method = 2;
		}
		else if (strcmp(argv[i], "-train") == 0)
		{
			i++;
			train_pic_path = argv[i];
		}
		else if (strcmp(argv[i], "-predict_anno"))
		{
			i++;
			predict_anno_path = argv[i];
		}
		else if (strcmp(argv[i], "-predict") == 0)
		{
			i++;
			classification_pic_path = argv[i];
		}
		else
			break;
	}

	if (i < argc)
		help();

	switch (method)
	{
	case 1:
		svm_classifier(train_pic_path, annotation_path, classification_pic_path, predict_anno_path, filename_to_save, filename_to_load);
		break;
	case 2:
		boost_classifier(train_pic_path, annotation_path, classification_pic_path, predict_anno_path, filename_to_save, filename_to_load);
		break;
	default:
		break;
	}

	return 0;
}

int main(int argc, char *argv[]){
	run(argc, argv);

	system("Pause");
}