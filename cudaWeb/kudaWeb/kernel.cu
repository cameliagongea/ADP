#include "imgproc.cuh"

int main(int argc, char** argv)
{																																																																																							
	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("src", WINDOW_AUTOSIZE);
	string filename = "photo.png";
	Mat src, dst;
	src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	src.convertTo(src, CV_8UC1);
	clock_t start, stop;
	start = clock();
	medianGPU_opti(src, dst);
	imwrite("Myphoto.jpg", dst);

	stop = clock();
	cout << "All cost time is " << static_cast<double>(stop - start) * 1000 / CLOCKS_PER_SEC << " ms" << endl;
	dst = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	return 0;
}
