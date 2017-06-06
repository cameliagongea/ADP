#include<stdio.h>
#include<stdlib.h>
#include <time.h>

// Number of vertices in the graph
#define V 6

/* Define Infinite as a large enough value. This value will be used
for vertices not connected to each other */
#define INFINITY 99999

// A function to print the solution matrix
void printSolution(int dist[][V]);

// Solves the all-pairs shortest path problem using Floyd Warshall algorithm
void floydWarshall(int graph[][V])
{
	/* dist[][] will be the output matrix that will finally have the shortest
	distances between every pair of vertices */
	int dist[V][V], i, j, k;

	/* Initialize the solution matrix same as input graph matrix. Or
	we can say the initial values of shortest distances are based
	on shortest paths considering no intermediate vertex. */
	for (i = 0; i < V; i++)
		for (j = 0; j < V; j++)
			dist[i][j] = graph[i][j];

	/* Add all vertices one by one to the set of intermediate vertices.
	---> Before start of a iteration, we have shortest distances between all
	pairs of vertices such that the shortest distances consider only the
	vertices in set {0, 1, 2, .. k-1} as intermediate vertices.
	----> After the end of a iteration, vertex no. k is added to the set of
	intermediate vertices and the set becomes {0, 1, 2, .. k} */
	for (k = 0; k < V; k++)
	{
		// Pick all vertices as source one by one
		for (i = 0; i < V; i++)
		{
			// Pick all vertices as destination for the
			// above picked source
			for (j = 0; j < V; j++)
			{
				// If vertex k is on the shortest path from
				// i to j, then update the value of dist[i][j]
				if (dist[i][k] + dist[k][j] < dist[i][j])
					dist[i][j] = dist[i][k] + dist[k][j];
			}
		}
	}

	// Print the shortest distance matrix
	printSolution(dist);
}

void printSolution(int dist[][V])
{
	int i, j;
	printf("Following matrix shows the shortest distances"
		" between every pair of vertices \n");
	for (i = 0; i < V; i++)
	{
		for (j = 0; j < V; j++)
		{
			if (dist[i][j] == INFINITY)
				printf("%7s", "INF");
			else
				printf("%7d", dist[i][j]);
		}
		printf("\n");
	}
}

// driver program to test above function
int main()
{

	int graph[V][V] = {
			{ 0, 3, 5, 1, INFINITY, INFINITY },
			{ 3, 0, INFINITY, INFINITY, 9, INFINITY },
			{ 5, INFINITY, 0, 7, 7, 1 },
			{ 1, INFINITY, 7, 0, INFINITY, 4 },
			{ INFINITY, 9, 7, INFINITY, 0, INFINITY },
			{ INFINITY, INFINITY, 1, 4, INFINITY, 0 }
	};
	clock_t start, stop;
	start = clock();

	// Print the solution
	floydWarshall(graph);

	stop = clock();
	printf("\nAll cost time is: %d", (stop - start) * 1000 / CLOCKS_PER_SEC);
	printf("ms\n");

	system("pause");
	return 0;
}