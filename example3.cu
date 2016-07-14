#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include "ParaLanczos.h"
 
// example of full warp occupancy
int main()
{


//	This is an arbitrary positive-symmetric 8 x 8 matrix we will use for testing.
	double g[] = {	11.6150,   12.6022,	2.7449,    0.6018,    5.7419,   -0.8989,    6.2368,    1.4880, 		\
			12.6022,   32.5907, 6.8222,    3.8457,    3.8868,   -1.3879,    7.2553,    1.3472, 		\
			2.7449,    6.8222,	11.3095,   1.9900,   -3.8608,    0.9067,    3.4324,    3.3344, 		\
			0.6018,    3.8457,	1.9900,    4.7412,   -0.0663,   -1.2597,   -0.2133,   -1.6123, 		\
			5.7419,    3.8868,	-3.8608,   -0.0663,   18.0182,   -1.7439,   -1.5991,    1.2881, 	\
			-0.8989,   -1.3879,	0.9067,   -1.2597,   -1.7439,    1.4386,    1.4156,    1.8073, 		\
			6.2368,    7.2553,	3.4324,   -0.2133,   -1.5991,    1.4156,    8.2406,    2.8208, 		\
			1.4880,    1.3472,	3.3344,   -1.6123,    1.2881,    1.8073,    2.8208,    4.7334 	};


	int n = 32;
	int matsize = 8;
	int minor = 8;
	int subIndexLength = 1 * 32;

	double Det;
	double *Eigenvalues;
	int *subIndex;

	subIndex = (int *) malloc(sizeof(int) * 32*8);

//	We set up an submatrix index whose eigenvalues we will extract on the GPU.

	for(int i = 0; i < subIndexLength; i += minor)
	{
		for(int k = 0; k < minor; k++)
		{
			subIndex[i+k] = k;
		}
	}

	Eigenvalues = (double *) malloc(sizeof(double)*subIndexLength);

//	This program loads the matrix above on the GPU, and then checks the above submatrix.
	calculate_subeigs3(g, subIndexLength, matsize, Eigenvalues, &Det, subIndex, minor);



}


