#include "imgproc.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <texture_fetch_functions.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

const int block_x = 16;
const int block_y = 16;
const int blocksize = 64;
__constant__ int width;
__constant__ int height;

texture<uchar, 1, cudaReadModeElementType> texRef;

__device__  void isort(uchar* lhs, int N)
{
	int i, j;
	uchar temp;
	for (i = 1; i < N; ++i)
	{
		j = i - 1;
		temp = lhs[i];
		while (j > -1 && lhs[j] > temp)
		{
			lhs[j + 1] = lhs[j];
			--j;
		}
		lhs[j + 1] = temp;
	}
}
__global__ void cuda_median_fil(uchar* src, uchar* dst)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int index_x = x + blockIdx.x * (blockDim.x - 2);
	int index_y = y + blockIdx.y * blockDim.y + 1;
	if (index_x < width && index_y < height - 1)
	{
		__shared__ uchar temp[block_y * 3][block_x];
		int top, mid, down, change;
		top = tex1Dfetch(texRef, index_x + (index_y - 1) * width);
		mid = tex1Dfetch(texRef, index_x + index_y * width);
		down = tex1Dfetch(texRef, index_x + (index_y + 1) * width);
		if (top < mid)
		{
			change = mid;
			mid = top;
			top = change;
		}
		if (top < down)
		{
			change = down;
			down = top;
			top = change;
		}
		if (mid < down)
		{
			change = down;
			down = mid;
			mid = change;
		}
		int index = 3 * y;
		temp[index][x] = top;
		temp[index + 1][x] = mid;
		temp[index + 2][x] = down;
		__syncthreads();
		if (x > 0 && x < block_x - 1)
		{
			uchar box[3][3];
			for (int i = 0; i < 3; ++i)
			{
				for (int j = -1; j < 2; ++j)
				{
					box[i][j] = temp[index + i][x + j];
				}
			}
			for (int i = 0; i < 3; ++i)
			{
				isort(&box[i][0], 3);
			}

			if (box[0][0] < box[1][1])
			{
				change = box[0][0];
				box[0][0] = box[1][1];
				box[1][1] = change;
			}
			if (box[0][0] < box[2][2])
			{
				change = box[0][0];
				box[0][0] = box[2][2];
				box[2][2] = change;
			}
			if (box[1][1] < box[2][2])
				dst[index_x + index_y * width] = box[2][2];
			else
				dst[index_x + index_y * width] = box[1][1];
		}
	}

}
void medianGPU(Mat src, Mat& dst)
{
	if (src.type() != CV_8UC1)
	{
		src.convertTo(src, CV_8UC1);
	}
	copyMakeBorder(src, src, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	int size = src.rows * src.cols;
	int count = size * sizeof(uchar);
	uchar* dsrc, *ddst, *hdst;
	hdst = (uchar*)malloc(count);
	memset(hdst, 0, count);
	if (!hdst)
	{
		cerr << "Host memory allocated failed!" << endl;
		return;
	}
	//allocate device memory
	cudaMalloc((void**)&dsrc, count);
	cudaMalloc((void**)&ddst, count);
	//copy host to device
	cudaMemcpy(dsrc, (uchar*)(src.ptr<uchar>(0)), count, cudaMemcpyHostToDevice);
	//width and height
	cudaMemcpyToSymbol((const void*)&width, (const void*)&src.cols, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol((const void*)&height, (const void*)&src.rows, sizeof(int), 0, cudaMemcpyHostToDevice);
	//kernel function to do median filter
	dim3 block(blocksize);
	dim3 grid(src.rows * src.cols / block.x + 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cuda_median_fil << <grid, block >> >(dsrc, ddst);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "Elapsed time is " << elapsed_time << " ms" << endl;
	// Check if kernel execution generated and error
	//memcpy from device to host
	cudaMemcpy(hdst, ddst, count, cudaMemcpyDeviceToHost);
	//copy to dst
	dst = Mat_<uchar>(src.rows, src.cols, hdst).clone();
	//release memory
	cudaFree(dsrc);
	cudaFree(ddst);
	free(hdst);
}

void medianGPU_opti(Mat src, Mat& dst)
{	
	if (src.type() != CV_8UC1)
	{
		src.convertTo(src, CV_8UC1);
	}
	copyMakeBorder(src, src, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	int size = src.rows * src.cols;
	int count = size * sizeof(uchar);
	uchar* dsrc, *ddst, *hdst;
	hdst = (uchar*)malloc(count);
	memset(hdst, 0, count);
	if (!hdst)
	{
		cerr << "Host memory allocated failed!" << endl;
		return;
	}
	//allocate device memory
	cudaMalloc((void**)&dsrc, count);
	cudaMalloc((void**)&ddst, count);
	//copy host to device
	cudaMemcpy(dsrc, src.data, count, cudaMemcpyHostToDevice);
	//bind to texture reference
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
	cudaBindTexture(0, &texRef, dsrc, &channelDesc, count);
	//width and height
	cudaMemcpyToSymbol((const void*)&width, (const void*)&src.cols, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol((const void*)&height, (const void*)&src.rows, sizeof(int), 0, cudaMemcpyHostToDevice);
	//kernel function to do median filter
	dim3 block(block_x, block_y);
	dim3 grid((src.cols + block_x - 2) / (block_x - 2), (src.rows + block_y - 2) / block_y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cuda_median_fil << <grid, block >> >(dsrc, ddst);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "elapsed time is " << elapsed_time << " ms" << endl;
	//unbind memory from texturereference
	cudaUnbindTexture(&texRef);
	//memcpy from device to host
	cudaMemcpy(hdst, ddst, count, cudaMemcpyDeviceToHost);
	//copy to dst
	dst = Mat_<uchar>(src.rows, src.cols, hdst).clone();
	//release memory
	cudaFree(dsrc);
	cudaFree(ddst);
	free(hdst);
}