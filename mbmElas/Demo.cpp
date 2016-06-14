#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h> 
#include "MBMElas\MBMElas.h"
#include <time.h>


using namespace std;
using namespace cv;


int main()
{
    //load images
	char strLeft[100];
	char strRight[100];
	char strDepth[100];
	
	for (int start=6; start<15; start++)
	{
		sprintf(strLeft, "%s%d%s", "D:/testPicture/images/rectify/dst_left", start, ".jpg");
		sprintf(strRight, "%s%d%s", "D:/testPicture/images/rectify/dst_right", start, ".jpg");
		sprintf(strDepth, "%s%d%s", "D:/testPicture/images/rectify/depth", start, ".jpg");

		/*sprintf(strLeft, "%s%d%s", "D:/testPicture/images/removeSync/linear/left_dst", start, ".jpg");
		sprintf(strRight, "%s%d%s", "D:/testPicture/images/removeSync/linear/right_dst", start, ".jpg");
		sprintf(strDepth, "%s%d%s", "D:/testPicture/images/removeSync/linear/depth", start, ".jpg");*/
		
		/*sprintf(strLeft, "%s%d%s", "D:/testPicture/images/newRectify/newPc/rectify_ckt_0", start, "_mainRGB.jpg");
		sprintf(strRight, "%s%d%s", "D:/testPicture/images/newRectify/newPc/rectify_ckt_0", start, "_subRGB.jpg");
		sprintf(strDepth, "%s%d%s", "D:/testPicture/images/newRectify/newPc/depth", start, ".jpg");*/

		//sprintf(strLeft, "%s%d%s", "D:/testPicture/images/newRectify/rectify_ckt_0", start, "_mainRGB.jpg");
		//sprintf(strRight, "%s%d%s", "D:/testPicture/images/newRectify/rectify_ckt_0", start, "_subRGB.jpg");
		//sprintf(strDepth, "%s%d%s", "D:/testPicture/images/newRectify/depth", start, ".jpg");

	  /*sprintf(strLeft, "%s%d%s", "D:/testPicture/images/newRectify/linear/left_dst", start, ".jpg");
	  sprintf(strRight, "%s%d%s", "D:/testPicture/images/newRectify/linear/right_dst", start, ".jpg");
	  sprintf(strDepth, "%s%d%s", "D:/testPicture/images/newRectify/linear/depth", start, ".jpg");*/

		/*if(start<10){
			sprintf(strLeft, "%s%d%s", "D:/testPicture/pone3/images/rectify_ckt_0", start, "_mainRGB.jpg");
			sprintf(strRight, "%s%d%s", "D:/testPicture/pone3/images/rectify_ckt_0", start, "_subRGB.jpg");
			sprintf(strDepth, "%s%d%s", "D:/testPicture/pone3/MBMDepth02/depth0", start, ".jpg");
		}
		else{
			sprintf(strLeft, "%s%d%s", "D:/testPicture/pone3/images/rectify_ckt_", start, "_mainRGB.jpg");
			sprintf(strRight, "%s%d%s", "D:/testPicture/pone3/images/rectify_ckt_", start, "_subRGB.jpg");
			sprintf(strDepth, "%s%d%s", "D:/testPicture/pone3/MBMDepth02/depth", start, ".jpg");
		}*/

		/*if(start<10){
			sprintf(strLeft, "%s%d%s", "D:/testPicture/QTek/images/new/rectify_ckt_0", start, "_mainRGB.jpg");
			sprintf(strRight, "%s%d%s", "D:/testPicture/QTek/images/new/rectify_ckt_0", start, "_subRGB.jpg");
			sprintf(strDepth, "%s%d%s", "D:/testPicture/QTek/images/MBMDepth/depth0", start, ".jpg");
		}
		else{
			sprintf(strLeft, "%s%d%s", "D:/testPicture/QTek/images/new/rectify_ckt_", start, "_mainRGB.jpg");
			sprintf(strRight, "%s%d%s", "D:/testPicture/QTek/images/new/rectify_ckt_", start, "_subRGB.jpg");
			sprintf(strDepth, "%s%d%s", "D:/testPicture/QTek/images/MBMDepth/depth", start, ".jpg");
		}*/
		double duration = static_cast<double>(getTickCount());

		Mat lImg = imread( strLeft, IMREAD_COLOR);
		Mat rImg = imread( strRight, IMREAD_COLOR);
		if( !lImg.data || !rImg.data ) {
			printf( "Error: can not open image\n" );
			continue;
		}

		//preprocessing (0.09s)
		Mat left, right;
		cvtColor(lImg, left, CV_BGR2GRAY);  //0.085s
		cvtColor(rImg, right, CV_BGR2GRAY);

		//medianBlur(left, left, 3);
		//medianBlur(right, right, 5);

		GaussianBlur(left, left, Size(3, 3), 0.9);  //0.05s
		GaussianBlur(right, right, Size(3, 3), 0.9);

		//init MBMElas object
		MBMElas::parameters param;
		MBMElas melas(param);

		// get image width and height
		int32_t width  = left.cols;
		int32_t height = left.rows;
		
		//init disparity 
		Mat leftDisparity = Mat::zeros(left.size(), CV_8UC1);
		Mat rightDisparity = Mat::zeros(left.size(), CV_8UC1);
		Mat maxMncc = Mat::ones(left.size(), CV_32FC1) * (-1);

		//double duration = static_cast<double>(getTickCount());

		//get left and right disparity (1s)
		melas.getDisparity(left, right, leftDisparity, rightDisparity, maxMncc, lImg, rImg);  
		/*normalize(leftDisparity, leftDisparity, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("leftDisparity", leftDisparity);
		waitKey(0);*/

	
		//validateDisparity
		//filterSpeckles

		//Mat temp1, temp2;
		//normalize(leftDisparity, temp1, 0, 255, NORM_MINMAX, CV_8UC1);
		//normalize(rightDisparity, temp2, 0, 255, NORM_MINMAX, CV_8UC1);

		//imshow("leftDisparity", temp1);
		//imshow("rightDisparity", temp2);
		//waitKey(0);
		//continue;
		//imwrite("strDepth.jpg", leftDisparity);
		// allocate memory for disparity images
		const int32_t dims[3] = {width,height,width}; // bytes per line = width
		float* D1_data = (float*)malloc(width*height*sizeof(float));
		float* D2_data = (float*)malloc(width*height*sizeof(float));
		float* D_data = (float*)malloc(width*height*sizeof(float));

		Mat initLeftDisparity, initRightDisparity;
		leftDisparity.convertTo(initLeftDisparity, CV_32FC1);
		rightDisparity.convertTo(initRightDisparity, CV_32FC1);

		memcpy(D1_data, initLeftDisparity.data, width * height * sizeof(float));
		memcpy(D2_data, initRightDisparity.data, width * height * sizeof(float));
		
		// refine left disparity (spend 0.053s)
		melas.refineLeftDisparity(D1_data,D2_data,D_data, dims, left, maxMncc, leftDisparity);

		duration = static_cast<double>(getTickCount())-duration;
		duration /= cv::getTickFrequency(); 
		cout << "duration = " <<duration * 1000 << endl;
		/*duration = static_cast<double>(getTickCount())-duration;
		duration /= cv::getTickFrequency(); 
		cout << "duration2 = " <<duration << endl;*/

		Mat Dis = Mat(left.rows, left.cols, CV_32FC1, D_data);
		Mat temp;
		Dis.convertTo(temp, CV_8UC1);

		normalize(temp, temp, 0, 255, NORM_MINMAX, CV_8UC1);
		
		//dtFilter(lImg, temp, temp, 120, 0.8*255, DTF_NC, 3);
		imwrite(strDepth, temp);
		//normalize(temp, temp, 0, 255, NORM_MINMAX, CV_8UC1);
		//imshow("src", left);
        imshow("temp", temp);
		waitKey(0);

	}
	return 0;
}

/*
double duration = static_cast<double>(getTickCount());
duration = static_cast<double>(getTickCount())-duration;
duration /= cv::getTickFrequency(); 
cout << duration << endl;
*/