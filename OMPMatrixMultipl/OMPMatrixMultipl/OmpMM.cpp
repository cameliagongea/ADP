#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 62                 
#define NCA 15                 
#define NCB 7                

int main(int argc, char *argv[])
{
	int	threadId, nthreads, i, j, k, chunk;
	double	a[NRA][NCA],
		b[NCA][NCB],
		c[NRA][NCB];

	chunk = 10;

#pragma omp parallel shared(a,b,c,nthreads,chunk) private(threadId,i,j,k)
	{
		//ia id-ul primului thread
		threadId = omp_get_thread_num();
		if (threadId == 0)
		{
			nthreads = omp_get_num_threads();
			printf("Nr de threaduri este %d\n", nthreads);
		}

#pragma omp for schedule (static, chunk) 
		for (i = 0; i<NRA; i++)
		for (j = 0; j<NCA; j++)
			a[i][j] = i + j;
#pragma omp for schedule (static, chunk)
		for (i = 0; i<NCA; i++)
		for (j = 0; j<NCB; j++)
			b[i][j] = i*j;
#pragma omp for schedule (static, chunk)
		for (i = 0; i<NRA; i++)
		for (j = 0; j<NCB; j++)
			c[i][j] = 0;

#pragma omp for schedule (static, chunk)
		for (i = 0; i<NRA; i++)
		{
			printf("Thread=%d a inmultit randul %d\n", omp_get_thread_num(), i);
			for (j = 0; j<NCB; j++)
			for (k = 0; k<NCA; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}

	for (i = 0; i<NRA; i++)
		//{
	for (j = 0; j<NCB; j++)
		//printf("%6.2f   ", c[i][j]);
		//printf("\n");
		//}
		system("pause");
}