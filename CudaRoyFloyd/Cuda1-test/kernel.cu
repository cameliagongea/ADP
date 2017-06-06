#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INF -1;

void Generatormatrixrix(int *matrix, const size_t VerticesNbr)
{
	for (int i = 0; i < VerticesNbr*VerticesNbr; i++)
	{
		
		if (rand() % 3 == 0)
		{
			matrix[i] = INF;
			
		}
		else
		{
			matrix[i] = rand() % 32;
		}
	}
	for (int i = 0; i < VerticesNbr; i++)
		matrix[i*VerticesNbr + i] = 0;
}

__device__
int Minimum(int a, int b) { return a < b ? a : b; }

__global__
void RoyFloyd(int* matrix, int k, int VerticesNbr) {
	
		int i = blockIdx.x;
		int j = threadIdx.x;

		if (matrix[i*VerticesNbr + k] != -1 || matrix[k*VerticesNbr + j] != -1)
		{
			matrix[i*j] = Minimum(matrix[i*VerticesNbr + k] + matrix[k*VerticesNbr + j], matrix[i*j]);
		}
}

void ExecuteRoyFloyd(int* matrix, int VerticesNbr, int thread_per_block){
	int* cuda_matrix;
	int size = sizeof(int)* VerticesNbr * VerticesNbr;
	cudaMalloc((void**)&cuda_matrix, size);
	cudaMemcpy(cuda_matrix, matrix, size, cudaMemcpyHostToDevice);
	int num_block = ceil(1.0*VerticesNbr*VerticesNbr / (thread_per_block));
	for (int k = 0; k < VerticesNbr; ++k)
	{
		RoyFloyd << <num_block, (thread_per_block) >> >(cuda_matrix, k, VerticesNbr);
	} 
	cudaMemcpy(matrix, cuda_matrix, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_matrix);
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	int thread_per_block = 256, i, j;
	//generate a random matrixrix.
	size_t VerticesNbr = 10;
	int *matrix = (int*)malloc(sizeof(int)*VerticesNbr*VerticesNbr);
	Generatormatrixrix(matrix, VerticesNbr);

	//compute your results
	int *result = (int*)malloc(sizeof(int)*VerticesNbr*VerticesNbr);
	memcpy(result, matrix, sizeof(int)*VerticesNbr*VerticesNbr);
	//replace by parallel algorithm
	ExecuteRoyFloyd(result, VerticesNbr, thread_per_block);

	for (i = 0; i < VerticesNbr; i++)
	{
		for (j = 0; j < VerticesNbr; j++)
			printf("distance[%d][%d] = %d \n", i, j, result[i*VerticesNbr + j]);
	}

	system("pause");

}