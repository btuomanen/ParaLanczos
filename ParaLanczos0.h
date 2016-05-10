/* This file is part of the ParaLanczos Library 

  Copyright (c) 2015-2016 Brian Tuomanen 

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  See the file LICENSE included with this distribution for more
  information. */

#define FLT_EPSILON 	1.192093e-07
#define DBL_EPSILON 	2.220446e-16
#define LDBL_EPSILON  	1.084202e-19

#define MAX_ITER_EIG	64
#define EIG_PRECISION	DBL_EPSILON

// some useful macros follow:

#define _POW2(x)	(x * x)
#define _ABS(x)  	(x < 0 ? -x : x )
#define _MAX(x,y)  	(x > y ? x : y )
#define _MIN(x,y) 	(x > y ? y : x )
#define _ABS(x)		(x < 0 ? -x : x )
#define _PYTHAG(a,b)	sqrt(a*a + b*b)
// zero is 'positive' sign in this macro
#define _SIGN(x)	(x < 0 ? -1 : 1)
// zero is 'zero'-sign in the following macro
#define _SIGN0(x)	(x < 0 ? -1 : (x == 0 ? 0 : 1))
#define _EPSILON	DBL_EPSILON


#define CHECK(call)     { 								\
    const cudaError_t error = call; 							\
    if (error != cudaSuccess) 								\
			{  								\
    			fprintf(stderr, "Error: %s:%d, ", __FILE__,__LINE__);          	\
    			fprintf(stderr, "code: %d, reason: %s\n", error, 		\
    			cudaGetErrorString(error)); 					\
			} 								\
}


#define MAT3D(A1, I1, J1, K1)		A1[ (int) ( ( (int) K1 ) * ((int) blockDim.x)  * ((int) blockDim.y) ) +  (int) ( ((int) J1)  + ( (int) blockDim.y) * ((int) I1) ) ]


// Unnecessary, but for preserving sanity

// yes, this is an overkill, but I always write these backwards...

#define th_i	((int) threadIdx.x)
#define th_j	((int) threadIdx.y)
#define th_k	((int) threadIdx.z)
#define	thx	th_i
#define thy	th_j
#define thz	th_k
#define thi	th_i
#define thj	th_j
#define thk	th_k
#define th_null (((int)threadIdx.x == 0) && ((int)threadIdx.y == 0) && ((int)threadIdx.z == 0))
#define th0	th_null
#define thnull	th_null
#define i_dim	((int) blockDim.x)
#define j_dim	((int) blockDim.y)
#define k_dim	((int) blockDim.z)
#define idim	i_dim
#define jdim	j_dim
#define kdim	k_dim
#define dimi	idim
#define dimj	jdim
#define	dimk	kdim
#define dimx	i_dim
#define dimy	j_dim
#define dimz	k_dim
#define xdim	dimx
#define ydim	dimy
#define zdim	dimz


typedef struct SubSeq
{
	int *Index;
	int len;
	double Lower;	// Lower bound A (really a mean)
	double Lstd;	// (since several lower bounds are computed--this is the mean.)
	double Upper;	// Upper bound (a mean)
	double Ustd;	// standard deviation
} subseq;




typedef struct subMatrix4
{
	// structure for 4 x 4 matrix

	double vec[4];
	double vecw[4];
	double vec_[4];

	double mat[4 * 4];  
	double w;
	double a;
	double b;

} sms4;


