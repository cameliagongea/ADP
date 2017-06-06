#include <ctime>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda.h>
#include <device_launch_parameters.h>
using namespace std;
using namespace cv;

void medianGPU(Mat src, Mat& dst);

void medianGPU_opti(Mat src, Mat& dst);
