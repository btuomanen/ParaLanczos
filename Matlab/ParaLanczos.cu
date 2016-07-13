/* This file is part of the ParaLanczos Library 

  Copyright (c) 2015-2016 Brian Tuomanen 

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  See the file LICENSE included with this distribution for more
  information. */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#include "ParaLanczos0.h"
#include "ParaLanczos.h"

/* This function (originally serial CPU ) is from EISPACK. */

__device__ int TQLRAT ( int n, double d[], double e2[] )
{
  double b;
  double c;
  double f;
  double g;
  double h;
  int i;
  int ierr;
  int ii;
  int j;
  int l;
  int l1;
  int m;
  int mml;
  double p;
  double r;
  double s;
  double t;

  ierr = 0;

  if ( n == 1 )
  {
    return ierr;
  }

  for ( i = 1; i < n; i++ )
  {
    e2[i-1] = e2[i];
  }

  f = 0.0;
  t = 0.0;
  e2[n-1] = 0.0;

  for ( l = 0; l < n; l++ )
  {
     j = 0;
     h = _ABS ( d[l] ) + sqrt ( e2[l] );

     if ( t <= h )
     {
       t = h;
       b = _ABS ( t ) * _EPSILON;
       c = b * b;
     }
/*
  Look for small squared sub-diagonal element.
*/
    for ( m = l; m < n; m++ )
    {
      if ( e2[m] <= c )
      {
        break;
      }
    }

    if ( m != l )
    {
      for ( ; ; )
      {
        if ( 30 <= j )
        {
          ierr = l + 1;
          return ierr;
        }

        j = j + 1;
/*
  Form shift.
*/
        l1 = l + 1;
        s = sqrt ( e2[l] );
        g = d[l];
        p = ( d[l1] - g ) / ( 2.0 * s );
        r = _PYTHAG ( p, 1.0 );
        d[l] = s / ( p + _ABS ( r ) * _SIGN ( p ) );
        h = g - d[l];
        for ( i = l1; i < n; i++ )
        {
          d[i] = d[i] - h;
        }
        f = f + h;
/*
  Rational QL transformation.
*/
        g = d[m];
        if ( g == 0.0 )
        {
          g = b;
        }

        h = g;
        s = 0.0;
        mml = m - l;

        for ( ii = 1; ii <= mml; ii++ )
        {
          i = m - ii;
          p = g * h;
          r = p + e2[i];
          e2[i+1] = s * r;
          s = e2[i] / r;
          d[i+1] = h + s * ( h + d[i] );
          g = d[i] - e2[i] / g;
          if ( g == 0.0 )
          {
            g = b;
          }
          h = g * p / r;
        }
        e2[l] = s * g;
        d[l] = h;
/*
  Guard against underflow in convergence test.
*/
        if ( h == 0.0 )
        {
          break;
        }

        if ( _ABS ( e2[l] ) <= _ABS ( c / h ) )
        {
          break;
        }

        e2[l] = h * e2[l];

        if ( e2[l] == 0.0 )
        {
          break;
        }
      }
    }

    p = d[l] + f;
/*
  Order the eigenvalues.
*/
    for ( i = l; 0 <= i; i-- )
    {
      if ( i == 0 )
      {
        d[i] = p;
        break;
      }
      else if ( d[i-1] <= p )
      {
        d[i] = p;
        break;
      }
      d[i] = d[i-1];
    }
  }

  return ierr;
}

// Extend the shuffle commands to 64 bit doubles with the magic of inline ptx assembly language.

__device__ __inline__ double shfl64(double val, int lane, int warpSize)
{
	int low, high;
	// the 'volatile' keeps the commands in place.`

	// bitwise-parition the 64 bit double into two 32 bit registers
	asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(low),"=r"(high):"d"(val));

	low = __shfl(low,lane,warpSize);
	high = __shfl(high,lane,warpSize);

	asm volatile("mov.b64 %0, {%1, %2}; ":"=d"(val):"r"(low),"r"(high));
	return val;
}

__device__ __inline__ double shfl_down64(double val, int lane, int warpSize)
{
	int low, high;
	// the 'volatile' keeps the commands in place.`

	// bitwise-parition the 64 bit double into two 32 bit registers
	asm volatile("mov.b64 {%0, %1}, %2;\n\t":"=r"(low),"=r"(high):"d"(val));

	low = __shfl_down(low,lane,warpSize);
	high = __shfl_down(high,lane,warpSize);

	asm volatile("mov.b64 %0, {%1, %2};\n\t":"=d"(val):"r"(low),"r"(high));
	return val;
}


// just to test the partitioning of a double in ptx assembly -- delete this some day.
__device__ void asm64test()
{
	double x = (double) 3.141619f;
	double y;
	int lo, hi;
	asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(lo),"=r"(hi):"d"(x));
	asm volatile("mov.b64 %0, {%1, %2};\n\t":"=d"(y):"r"(lo),"r"(hi));

	printf("x = %g, y = %g \n", x,y);

}

// determines determinant from tridiagonalization of symmetric matrix--b[0] is 0, of course.
// uses the method of continuants -- standard method.
//
//
// inputs: n, a, b
// outputs theta continuants into pointer  (used for calculating inverse)
__inline__ __device__ double DET_TRIDIAG (int n, double a[], double b[] )
{
	double fn_1, fn_2, fn;
	// use doubles to avoid numerical loss, convert back to double upon return
	int step;

	// initialize
	fn_1 = 1.0;
	fn_2 = 0.0;
	fn = 0.0;  // unnecessary




	for(step = 0; step  < n; step++)
	{
		fn = ( a[step])  * fn_1 - _POW2( b[step]) * fn_2;
		//theta[step] = (double) fn;

		fn_2 = fn_1;
		fn_1 = fn;
	}


	return(fn);

}

__inline__ __device__ void TRIDIAG_CONTINUANTS(int n, double a[], double b[], double theta[], double phi[])
{
	int i;

	theta[0] = 1.0;
	theta[1] = a[0];

	for(i=2; i < n + 1; i++)
	{
		theta[i] = a[i-1] * theta[i-1] - _POW2(b[i-1]) * theta[i - 2];
	}

	phi[n] = 1.0;
	phi[n-1]=a[n-1];

	for(i=n-2; i >= 0; i--)
	{
		phi[i] = a[i+1] * phi[i+1] - _POW2(b[i+1]) * phi[i + 2];
	}

	return;

}


// The Usani method for inverting a symmetric tridiagonal matrix.
//
__inline__ __device__ double INVERT_TRIDIAG_IJ(int n, double b[], double theta[], double phi[], int i, int j)
{
	int k,m; // index variables since i, j taken for matrix

	double temp=1.0;

	// swap values of i, j in this case.
	if( i > j )
	{
		m = j;
		j = i;
		i = m;
	}

	for(k=i; k <= j - 1 ; k++)
	{
		temp *= b[k+1];
	}

	temp *= ( theta[i]) * ( phi[j+1]);
	temp /= ( theta[n]);

	return( (i + j) % 2 == 0 ? temp : -temp);

}

// square-sums a vector
__inline__ __device__ double CALC_NORMSQ(double vec[],  int len)
{
	int i;
	double x = 0;

	for ( i = 0; i < len; i++)
		{ x += vec[i]*vec[i]; }

	return (x);
}


// matrix multiplication for a 4x4 matrix structure (sms4)
// this gives vec1 = sms4.mat * vec
__inline__ __device__ void EVAL_SMS4(sms4 &subgram4, double vec[], double vec1[])
{

#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		vec1[i] = (vec[0] * subgram4.mat[0 + 4 * i] + vec[1] * subgram4.mat[1 + 4 * i] + vec[2] * subgram4.mat[2 + 4 * i] + vec[3] * subgram4.mat[3 + 4 * i]);
	}

}

// adds, returns into vec1
__inline__ __device__ void  ADD_2_R4_VECTORS(double vec1[], double vec2[], double vecout[])
{

#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		vecout[i] = vec1[i] + vec2[i];
	}

}

/* DSG functions:  for computing Gram matrices */

__global__ void DSG_B(double *subgrams, double *Frame, int M, int N, int *subset, int n, int offset1)
{


	int k;
	double val;

	val = 0;

	for (k = 0; k < M; k++)
	{
		val += Frame[subset[th_i] + k*N + offset1] * Frame[subset[th_j] + k*N + offset1];
	}

	subgrams[th_j + th_i*n] = val;

	__syncthreads();

}



// Grammian, parallellized in one dimension.

__global__ void DSG_B1(double *subgrams, double *Frame, int M, int N, int *subset, int offset1)
{

	int k, j;
	double val[128];

	for (j = 0; j<dimx; j++)
	{

		val[j] = 0;

		for (k = 0; k<M; k++)
		{
			val[j] += Frame[subset[th_i] + k*N + offset1] * Frame[subset[j] + k*N + offset1];
		}

		subgrams[j + th_i*dimx] = val[j];
	}

	__syncthreads();
}

__global__ void SG_B1(double *subgrams, double *Frame, int card)
{

	int k, j;

	double *val;


	val = (double*) malloc(dimx*sizeof(double));

	__syncthreads();

	for (j = 0; j<dimx; j++)
	{

		val[j] = 0;

		for (k = 0; k<dimx; k++)
		{
			val[j] += Frame[th_i + k*card ] * Frame[j + k*card ];
		}

		subgrams[j + th_i*dimx] = val[j];
	}


	__syncthreads();
	free(val);
}

/* load from global memory into local matrix structure.*/
__inline__ __device__ void LOAD_SMS4X(sms4 &s4, double *mat, int n, int i, int j)
{
	int k, l;

	for(k = 0; k < 4; k++)
		for(l=0; l < 4; l++)
			s4.mat[k*4 + l] = mat[((i+k)*n + j + l)];


	s4.w = 0;
}

/* similar as above, loads from hash table. */
// check this some more. BT
__inline__ __device__ void LOAD_SMS4H(sms4 &s4, double *mat, int n, int i, int j, int * hash)
{
	int k, l;

	//for(k = 0; k < 4; k++)
		//printf("thread: %d, hash[%d] = %d, blockIdx.x = %d \n", thx, i+k   + blockIdx.x * 4 *dimx, hash[i+k   + blockIdx.x * 4 *dimx], blockIdx.x);

	for(k = 0; k < 4; k++)
		for(l=0; l < 4; l++)
			s4.mat[k*4 + l] = mat[(hash[i+k   + blockIdx.x * 4 *dimx])*n + hash[j+l  + blockIdx.x * 4 *dimx] ];


	s4.w = 0;
}

/* similar as above, loads from hash table. For the "full occupancy" kernel.*/
__inline__ __device__ void LOAD_SMS4HF(sms4 &s4, double *mat, int n, int i, int j, int * hash, int offset, int subn, int subwarp_thx, int dimx_subwarp)
{
	int k, l;


	for(k = 0; k < 4; k++)
	{
	//	printf("thx: %d, k: %d \n", thx, k);
		for(l=0; l < 4; l++)
		{
	//		printf("k, l = %d, %d thx: %d index1: %d, index2: %d  \n", k, l, thx, i+k   + offset * 4 * dimx_subwarp, j+l  + offset * 4* dimx_subwarp); 
//			printf("thx: %d LOAD_SMS$HF: (%d, %d) \n", thx, hash[i+k   + offset * 4 * dimx_subwarp], hash[j+l + offset*4*dimx_subwarp]);	

			s4.mat[k*4 + l] = mat[(hash[i+k   + offset * 4 * dimx_subwarp])*n + hash[j+l  + offset * 4* dimx_subwarp] ];

		}
	}

	s4.w = 0;
}

// dot product for R^4 vectors
__inline__ __device__ double DOT4(double vec1[], double vec2[])
{
	return( vec1[0]*vec2[0] + vec1[1] *vec2[1] + vec1[2] * vec2[2] + vec1[3] * vec2[3] );
}


/* No use for now */
/* bisection method -- characteristic polynomial */


/* take a look at these again... */
__device__ double BM_POLY(int i, double alpha, double beta, double x, double pm, double pmm)
{


	if( i == 0 )
		return(1);
	else if (i == 1)
		return(alpha - x);
	else if(i > 1)
		return( (alpha - x) * pm - _POW2(beta)  * pmm );
	else
		return( -1.0 );
}

/* uses above. from old paper */
__inline__ __device__ int EIGVALS_BELOW( const int iters, double alpha[], double beta[], double x)
{
	double temp;
	int i;


	int numBelow = 0;

	double pm = alpha[0] - x;
	double pmm = 0;

	// just the number of sign differences

	if( alpha[0] - x < 0 )
		numBelow = 1;
	else
		numBelow = 0;


#pragma unroll
	for(i = 2; i < iters; i++)
	{
		temp = BM_POLY(i, alpha[i], beta[i-1], x, pm, pmm);
		pmm = pm;
		pm = temp;

		if(pm * pmm < 0)
		       numBelow++;

	}

	return(numBelow);

}


// Performs tridiagonalization on a matrix, then calls TQLRAT to perform an eigenvalue decomposition.
__global__ void POS_SYM_MAT_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror)
{
	int i,j,k;
	sms4 smat[32];

	curandState cuState[4];


	__shared__ double Alpha[128];
	__shared__ double Beta[130];

	__shared__ double tempvec1[130];
	__shared__ double tempvec2[130];

	clock_t timeseed;

	if(th_null)
	{
		timeseed = clock();
	}

	timeseed = shfl64(timeseed,0,dimx);

	for(i=0; i < 4; i++)
		curand_init((unsigned long long ) timeseed, (unsigned long long) n, (unsigned long long) 4 * thx + i, &cuState[i]);


	for(i=0; i < 4; i++)
	{
		smat[0].vec[i] = curand_normal(&cuState[i]) ; //__sinf((double) (thx*4 + i)); //curand_uniform(dstate1) - 0.5;
	}


	// this is to compute the magnitude, for normalization.

	smat[0].b = DOT4(smat[0].vec,smat[0].vec);


	// multithreaded summation

	for(k=1; k < dimx; k *= 2)
	{
		smat[0].b += shfl_down64(smat[0].b, k , dimx );
	}


	smat[0].b = shfl64(smat[0].b,0, dimx);
	smat[0].b = sqrt(smat[0].b);	 //__fsqrt_rn(smat[0].b); // intrinsic


	if(th_null)
		printf(" \n \n ");


	//printf(" %f %f %f %f ", smat[0].vec[0], smat[0].vec[1], smat[0].vec[2], smat[0].vec[3]);

	/* if(th_null)
	{
		printf(" \n \n ");
		printf(" norm of vector: %f \n", smat[0].b);
	} */



	// (I).2.2 normalize random vector

	for(i=0; i<4; i++)
	{
		smat[0].vec[i] = smat[0].vec[i] / smat[0].b;
		smat[0].vecw[i] = 0.0;
		smat[0].vec_[i] = 0.0;
	}

	// (II) load 4x4 matrix stractures from main matrix

	for(i=0; i<dimx; i++)
	{
		LOAD_SMS4X(smat[i], mat, 4*dimx, i*4, thx*4);
	}

	for(i=1; i < dimx; i++)
	{
		for(j=0; j < 4; j++)
		{
			smat[i].vec[j] = smat[0].vec[j];
			tempvec1[1+j+thx*4] = smat[i].vec[j];
			smat[i].vec_[j] = 0.0;
			smat[i].vecw[j] = 0.0;
		}
	}

	tempvec1[0] = 0.0;
	tempvec1[n+1] = 0.0;

	tempvec1[0] = 0.0;
	tempvec1[dimx*4 + 2] = 0.0;

	tempvec2[0] = 0.0;
	tempvec2[dimx*4 + 2] = 0.0;



	// using l as the index

	Beta[0] = 0.0;
	int l;

	for(l=0; l < dimx*4; l++)
	{

		for(i=0; i< dimx ; i++)
		{
			EVAL_SMS4(smat[i], smat[0].vec, smat[i].vecw);
		}

		int k;

		for(i=0; i < dimx; i++)
		{


			for(k=1; k < dimx; k *= 2)
			{
				for(j=0; j < 4; j++)
					{ smat[i].vecw[j] += shfl_down64(smat[i].vecw[j], (unsigned int) k , dimx ); }
			}
		}


		for(i=0; i < dimx; i++)
		{
			for(j=0; j < 4; j++)
			{

				smat[i].vecw[j] = shfl64(smat[i].vecw[j], 0, dimx);
			}
		}


		smat[0].a = DOT4(smat[0].vec, smat[thx].vecw);

		for(k=1; k < dimx; k *= 2)
		{
			smat[0].a += shfl_down64(smat[0].a, (unsigned int) k, dimx);
		}

		smat[0].a =  shfl64(smat[0].a, 0, dimx); // propagate alpha value

		// create next w vector


		for(i=0; i < 4; i++)
		{
			smat[0].vecw[i] =  smat[thx].vecw[i] -  (smat[0].a * smat[0].vec[i] + Beta[l] * smat[0].vec_[i]);
		}
		smat[0].b = DOT4(smat[0].vecw,smat[0].vecw);

		for(k=1; k <  dimx; k *= 2)
		{
			smat[0].b += shfl_down64(smat[0].b, (unsigned int) k,  dimx);
		}

		smat[0].b = shfl64(smat[0].b,  0, dimx);
		smat[0].b = sqrt(smat[0].b); //__fsqrt_rn(smat[0].b); // intrinsic version


		// set vec_ before setting vec

		for(i=0; i < 4; i++)
		{
			smat[0].vec_[i] = smat[0].vec[i];
		}


		for(i=0; i <4 ; i++)
		{
			smat[0].vec[i] = smat[0].vecw[i] / smat[0].b;
		}

		// propagate down , so EVAL works. (unnecessary)
		for(j=1; j < dimx; j++)
		{
			for(i=0; i < 4; i++)
			{
				smat[j].vec[i] = smat[0].vec[i];
			}
		}


		if(th_null)
		{
			Beta[l+1] = smat[0].b;
			Alpha[l] = smat[0].a;

			//printf("beta %g , alpha %g \n", smat[0].b , smat[0].a);


		}
	}

	// pad end of shared array.
	if(th_null)
		Beta[n+1] = 0.0;


	double w=0.0;

	// pad beginning of shared array
	if(th_null)
	{
		tempvec1[0] = 0.0;
		tempvec2[0] = 0.0;
	}


	int jj=0;

	//for(jj=0; jj < 16; jj++)
	{

		for(i=0; i < 4; i++)
		{
			tempvec2[4*thx+1+i] =   curand_normal(&cuState[i]) - 0.5; // +  5*(Beta[4*thx+i] + Alpha[thx*4 + i] + Beta[4*thx+1]);
		}

		w = 0.0;

		for(i = 0; i < 4; i++)
		{
			w += _POW2(tempvec2[4*thx+1+i]);
		}

		for(k=1; k < dimx; k *= 2)
		{
			w += shfl_down64(w, (unsigned int) k , dimx );
		}

		w = shfl64(w, 0, dimx);  // this is || tempvec1 ||^2

		w = sqrt(w);


		for(int ii=0; ii < 10; ii++)
		{
			for(i=0; i < 4; i++)
			{
				tempvec1[4*thx+1+i] = tempvec2[4*thx+1+i] / w;
			}

			for(i=0; i < 4; i++)
			{
				tempvec2[4*thx+1+i] = Beta[4*thx+i]*tempvec1[4*thx+i] + Alpha[thx*4 + i]*tempvec1[4*thx+i+1] + Beta[4*thx+1]*tempvec1[4*thx+i+2];

			}

			w = 0.0f;

			for(i = 0; i < 4; i++)
			{
				w += _POW2(tempvec2[4*thx+1+i]);
			}

			for(k=1; k < dimx; k *= 2)
			{
				w += shfl_down64(w, (unsigned int) k , dimx );
			}

			w = shfl64(w, 0, dimx);  // this is || tempvec1 ||^2

			w = sqrt(w);
		}
	}

	// compute tridiagonal matrix eigenvalues, determinent in a single thread.  copy determinent and 'Eigenerror' to global memory.

	if(th_null)
	{
		int eigzz;

		for(i =0; i < 4*dimx+2; i++)
			Beta[i] *= Beta[i];

		eigzz = TQLRAT(4*dimx, Alpha, Beta);



		*DetMat = DET_TRIDIAG(n, Alpha, Beta);

		*Eigerror = eigzz;


	}

	// copy the eigenvalues computed above into the global memory.

	for(i = 0; i < 4*dimx; i++)
	{
		Eigenvalues[i + thx*4 ] = Alpha[i + thx*4];

	}

}


// same as above, use blockIdx to offset appropriate values
__global__ void POS_SYM_SUBMATRIX_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn)
{
	int i,j,k;
	sms4 smat[32];

	curandState cuState[4];


	__shared__ double Alpha[128];
	__shared__ double Beta[130];

	__shared__ double tempvec1[130];
	__shared__ double tempvec2[130];

	clock_t timeseed;

	/*
	DetMat = &DetMat[blockIdx.x];
	Eigenvalues = &Eigenvalues[subn * blockIdx.x];
	Eigerror = &Eigerror[blockIdx.x];
	subIndex = &subIndex[subn*blockIdx.x];*/

	int subOffset, eigOffset, offset;

	offset = blockIdx.x;
	eigOffset = subn * blockIdx.x;
	subOffset = subn * blockIdx.x;

	if(th_null)
	{
		timeseed = clock();
	}

	timeseed = shfl64(timeseed,0,dimx);

	for(i=0; i < 4; i++)
		curand_init((unsigned long long ) timeseed, (unsigned long long) n, (unsigned long long) 4 * thx + i, &cuState[i]);


	for(i=0; i < 4; i++)
	{
		smat[0].vec[i] = curand_normal(&cuState[i]) ; //__sinf((double) (thx*4 + i)); //curand_uniform(dstate1) - 0.5;
	}


	// this is to compute the magnitude, for normalization.

	smat[0].b = DOT4(smat[0].vec,smat[0].vec);


	// multithreaded summation

	for(k=1; k < dimx; k *= 2)
	{
		smat[0].b += shfl_down64(smat[0].b, k , dimx );
	}


	smat[0].b = shfl64(smat[0].b,0, dimx);
	smat[0].b = sqrt(smat[0].b);	 //__fsqrt_rn(smat[0].b); // intrinsic



	// (I).2.2 normalize random vector

	for(i=0; i<4; i++)
	{
		smat[0].vec[i] = smat[0].vec[i] / smat[0].b;
		smat[0].vecw[i] = 0.0;
		smat[0].vec_[i] = 0.0;
	}

	// (II) load 4x4 matrix structures from main matrix

	for(i=0; i<dimx; i++)
	{
		LOAD_SMS4H(smat[i], mat, n, i*4, thx*4,  subIndex );
	}

	for(i=1; i < dimx; i++)
	{
		for(j=0; j < 4; j++)
		{
			smat[i].vec[j] = smat[0].vec[j];
			tempvec1[1+j+thx*4] = smat[i].vec[j];
			smat[i].vec_[j] = 0.0;
			smat[i].vecw[j] = 0.0;
		}
	}

	tempvec1[0] = 0.0;
	tempvec1[n+1] = 0.0;

	tempvec1[0] = 0.0;
	tempvec1[dimx*4 + 2] = 0.0;

	tempvec2[0] = 0.0;
	tempvec2[dimx*4 + 2] = 0.0;



	// using l as the index

	Beta[0] = 0.0;
	int l;

	for(l=0; l < dimx*4; l++)
	{

		for(i=0; i< dimx ; i++)
		{
			EVAL_SMS4(smat[i], smat[0].vec, smat[i].vecw);
		}

		int k;

		for(i=0; i < dimx; i++)
		{


			for(k=1; k < dimx; k *= 2)
			{
				for(j=0; j < 4; j++)
					{ smat[i].vecw[j] += shfl_down64(smat[i].vecw[j], (unsigned int) k , dimx ); }
			}
		}


		for(i=0; i < dimx; i++)
		{
			for(j=0; j < 4; j++)
			{

				smat[i].vecw[j] = shfl64(smat[i].vecw[j], 0, dimx);
			}
		}


		smat[0].a = DOT4(smat[0].vec, smat[thx].vecw);

		for(k=1; k < dimx; k *= 2)
		{
			smat[0].a += shfl_down64(smat[0].a, (unsigned int) k, dimx);
		}

		smat[0].a =  shfl64(smat[0].a, 0, dimx); // propagate alpha value

		// create next w vector


		for(i=0; i < 4; i++)
		{
			smat[0].vecw[i] =  smat[thx].vecw[i] -  (smat[0].a * smat[0].vec[i] + Beta[l] * smat[0].vec_[i]);
		}
		smat[0].b = DOT4(smat[0].vecw,smat[0].vecw);

		for(k=1; k <  dimx; k *= 2)
		{
			smat[0].b += shfl_down64(smat[0].b, (unsigned int) k,  dimx);
		}

		smat[0].b = shfl64(smat[0].b,  0, dimx);
		smat[0].b = sqrt(smat[0].b); //__fsqrt_rn(smat[0].b);


		// set vec_ before setting vec

		for(i=0; i < 4; i++)
		{
			smat[0].vec_[i] = smat[0].vec[i];
		}


		for(i=0; i <4 ; i++)
		{
			smat[0].vec[i] = smat[0].vecw[i] / smat[0].b;
		}

		// propagate down , so EVAL works. (unnecessary)
		for(j=1; j < dimx; j++)
		{
			for(i=0; i < 4; i++)
			{
				smat[j].vec[i] = smat[0].vec[i];
			}
		}


		if(th_null)
		{
			Beta[l+1] = smat[0].b;
			Alpha[l] = smat[0].a;
		}
	}

	// pad end of shared array.
	if(th_null)
		Beta[n+1] = 0.0;


	double w=0.0;

	// pad beginning of shared array
	if(th_null)
	{
		tempvec1[0] = 0.0;
		tempvec2[0] = 0.0;
	}


	int jj=0;

	//for(jj=0; jj < 16; jj++)
	{

		for(i=0; i < 4; i++)
		{
			tempvec2[4*thx+1+i] =   curand_normal(&cuState[i]) - 0.5; // +  5*(Beta[4*thx+i] + Alpha[thx*4 + i] + Beta[4*thx+1]);
		}

		w = 0.0;

		for(i = 0; i < 4; i++)
		{
			w += _POW2(tempvec2[4*thx+1+i]);
		}

		for(k=1; k < dimx; k *= 2)
		{
			w += shfl_down64(w, (unsigned int) k , dimx );
		}

		w = shfl64(w, 0, dimx);  // this is || tempvec1 ||^2

		w = sqrt(w);


		for(int ii=0; ii < 10; ii++)
		{
			for(i=0; i < 4; i++)
			{
				tempvec1[4*thx+1+i] = tempvec2[4*thx+1+i] / w;
			}

			for(i=0; i < 4; i++)
			{
				tempvec2[4*thx+1+i] = Beta[4*thx+i]*tempvec1[4*thx+i] + Alpha[thx*4 + i]*tempvec1[4*thx+i+1] + Beta[4*thx+1]*tempvec1[4*thx+i+2];

			}

			w = 0.0f;

			for(i = 0; i < 4; i++)
			{
				w += _POW2(tempvec2[4*thx+1+i]);
			}

			for(k=1; k < dimx; k *= 2)
			{
				w += shfl_down64(w, (unsigned int) k , dimx );
			}

			w = shfl64(w, 0, dimx);  // this is || tempvec1 ||^2

			w = sqrt(w);
		}
	}

	// compute tridiagonal matrix eigenvalues, determinent in a single thread.  copy determinent and 'Eigenerror' to global memory.

	if(th_null)
	{
		int eigzz;

		for(i =0; i < 4*dimx+2; i++)
			Beta[i] *= Beta[i];

		eigzz = TQLRAT(4*dimx, Alpha, Beta);



		DetMat[blockIdx.x] = DET_TRIDIAG(n, Alpha, Beta);

		Eigerror[blockIdx.x] = eigzz;


	}

	// copy the eigenvalues computed above into the global memory.
// i think the error is in here somewhere....
	for(i = 0; i < 4; i++)
	{
		Eigenvalues[i + thx*4 + blockIdx.x * dimx * 4] = Alpha[i + thx*4];

	}
	__syncthreads();

}


__global__ void POS_SYM_SUBMATRIX_KER2(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn)
{
	int i,j,k;
	sms4 smat[32];

	curandState cuState[4];

// this is for subn up to size 32 .. make an entire new kernel for less subwarps
// if you have [32][165] or so for each __shared__ array, that is too much for a Maxwell-level card.
	__shared__ double Alpha[32][32];
	__shared__ double Beta[32][33];

	__shared__ double tempvec1[32][33];
	__shared__ double tempvec2[32][33];

	clock_t timeseed;
	
	int subwarp, num_subwarps, subwarp_null, subwarp_thx, threads_per_subwarp, dimx_subwarp;
	int subOffset, eigOffset, offset;

	threads_per_subwarp = subn / 4;

	dimx_subwarp = threads_per_subwarp;

	subwarp = thx / threads_per_subwarp;

	subwarp_null = subwarp * threads_per_subwarp;
	subwarp_thx = thx - subwarp_null;
	num_subwarps = 32 / (threads_per_subwarp); // number of subwarps in a warp


	offset = blockIdx.x * num_subwarps + subwarp;
	eigOffset = subn * offset;
	subOffset = subn * offset;

	printf("my thx is %d my subn is %d my subwarp is %d my dimx is %d my dimx_subwarp is %d \n", thx, subn, subwarp, dimx, dimx_subwarp);

	//printf("I. \n");

	timeseed = clock();


	for(i=0; i < 4; i++)
		curand_init((unsigned long long ) timeseed, (unsigned long long) n, (unsigned long long) 4 * subwarp_thx + i, &cuState[i]);


	for(i=0; i < 4; i++)
	{
		smat[0].vec[i] = curand_normal(&cuState[i]) ; //__sinf((double) (thx*4 + i)); //curand_uniform(dstate1) - 0.5;
	}

	// this is to compute the magnitude, for normalization.

	smat[0].b = DOT4(smat[0].vec,smat[0].vec);


	// multithreaded summation

	for(k=1; k < dimx_subwarp; k *= 2)
	{
		smat[0].b += shfl_down64(smat[0].b, /* subwarp_null + */ k , 32);
	}


	smat[0].b = shfl64(smat[0].b,subwarp_null, 32);
	smat[0].b = sqrt(smat[0].b);	 //__fsqrt_rn(smat[0].b); // intrinsic



	// (I).2.2 normalize random vector

	for(i=0; i<4; i++)
	{
		smat[0].vec[i] = smat[0].vec[i] / smat[0].b;
		smat[0].vecw[i] = 0.0;
		smat[0].vec_[i] = 0.0;
	}

	// (II) load 4x4 matrix structures from main matrix
	//printf("II.. \n");
	for(i=0; i<dimx_subwarp; i++)
	{
		//printf("II.1.1\n");
		LOAD_SMS4HF(smat[i], mat, n, i*4, subwarp_thx*4,  subIndex, offset, subn, subwarp_thx, dimx_subwarp );
		//printf("II.1.2\n");
	}

	// I think this is correct til here...
	//printf("II.2\n");

	for(i=1; i < dimx_subwarp; i++)
	{
		for(j=0; j < 4; j++)
		{
			// some of this whole operation may be redundant... the EVAL_SMS4 function
			// below only uses smat[0].vec

			//printf("II.2.1\n");
			smat[i].vec[j] = smat[0].vec[j];
			//printf("II.2.1.1\n");
			tempvec1[subwarp][1+j+subwarp_thx*4] = smat[i].vec[j];
			//printf("II.2.1.2\n");
			smat[i].vec_[j] = 0.0;
			smat[i].vecw[j] = 0.0;
			//printf("II.2.1.3\n");
		}
	}



	if(thx == subwarp_null)
	{

		tempvec1[subwarp][0] = 0.0;
		tempvec1[subwarp][n+1] = 0.0;

		tempvec1[subwarp][0] = 0.0;
		tempvec1[subwarp][subn*4 + 2] = 0.0;

		tempvec2[subwarp][0] = 0.0;
		tempvec2[subwarp][subn*4 + 2] = 0.0;

		Beta[subwarp][0] = 0.0;
	}

//	printf("11.3\n");
	int l; 

	for(l=0; l < dimx_subwarp*4; l++)
	{

		for(i=0; i< dimx_subwarp ; i++)
		{
			EVAL_SMS4(smat[i], smat[0].vec, smat[i].vecw);
		}

		int k;

		for(i=0; i < dimx_subwarp; i++)
		{


			for(k=1; k < dimx_subwarp; k *= 2)
			{
				for(j=0; j < 4; j++)
					{ smat[i].vecw[j] += shfl_down64(smat[i].vecw[j], (unsigned int) /* subwarp_null + */ k , 32); }
			}
		}


		for(i=0; i < dimx_subwarp; i++)
		{
			for(j=0; j < 4; j++)
			{

				smat[i].vecw[j] = shfl64(smat[i].vecw[j], subwarp_null, 32);
			}
		}


		smat[0].a = DOT4(smat[0].vec, smat[subwarp_thx].vecw);

		for(k=1; k < dimx_subwarp; k *= 2)
		{
			smat[0].a += shfl_down64(smat[0].a, (unsigned int) /*subwarp_null + */ k, 32);
		}

		smat[0].a =  shfl64(smat[0].a, subwarp_null, 32); // propagate alpha value

		// create next w vector


		for(i=0; i < 4; i++)
		{
			smat[0].vecw[i] =  smat[subwarp_thx].vecw[i] -  (smat[0].a * smat[0].vec[i] + Beta[subwarp][l] * smat[0].vec_[i]);
		}
		smat[0].b = DOT4(smat[0].vecw,smat[0].vecw);

		for(k=1; k <  dimx_subwarp; k *= 2)
		{
			smat[0].b += shfl_down64(smat[0].b, (unsigned int) /* subwarp_null + */ k,  32);
		}

		smat[0].b = shfl64(smat[0].b,  subwarp_null, 32);
		smat[0].b = sqrt(smat[0].b); //__fsqrt_rn(smat[0].b);


		// set vec_ before setting vec

		for(i=0; i < 4; i++)
		{
			smat[0].vec_[i] = smat[0].vec[i];
		}


		for(i=0; i <4 ; i++)
		{
			smat[0].vec[i] = smat[0].vecw[i] / smat[0].b;
		}

		// propagate down , so EVAL works. (unnecessary)
		for(j=1; j < dimx_subwarp; j++)
		{
			for(i=0; i < 4; i++)
			{
				smat[j].vec[i] = smat[0].vec[i];
			}
		}


		if(thx == subwarp_null)
		{
			if(th_null)
				printf("thx: %d, a : %g , b: %g ", thx, smat[0].a, smat[0].b);
			Beta[subwarp][l+1] = smat[0].b;
			Alpha[subwarp][l] = smat[0].a;
		}
	}

	// pad end of shared array.
	if(subwarp_null == thx)
		Beta[subwarp][l+1] = 0.0;


	double w=0.0;

	// pad beginning of shared array

	// BT:  this isn't reached... no thread matches up to subwarp_null ...
	if(subwarp_null == thx)
	{
		tempvec1[subwarp][0] = 0.0;
		tempvec2[subwarp][0] = 0.0;
	}


	int jj=0;

	//for(jj=0; jj < 16; jj++)
	{

		for(i=0; i < 4; i++)
		{
			tempvec2[subwarp][4*subwarp_thx+1+i] =   curand_normal(&cuState[i]); // +  5*(Beta[4*thx+i] + Alpha[thx*4 + i] + Beta[4*thx+1]);
		}

		w = 0.0;

		for(i = 0; i < 4; i++)
		{
			w += _POW2(tempvec2[subwarp][4*subwarp_thx+1+i]);
		}

		for(k=1; k < dimx_subwarp; k *= 2)
		{
			w += shfl_down64(w, (unsigned int) /* subwarp_null + */ k , 32);
		}

		w = shfl64(w, subwarp_null, 32);  // this is || tempvec1 ||^2

		w = sqrt(w);


		for(int ii=0; ii < 10; ii++)
		{
			for(i=0; i < 4; i++)
			{
				tempvec1[subwarp][4*subwarp_thx+1+i] = tempvec2[subwarp][4*subwarp_thx+1+i] / w;
			}

			for(i=0; i < 4; i++)
			{
				tempvec2[subwarp][4*subwarp_thx+1+i] = Beta[subwarp][4*subwarp_thx+i]*tempvec1[subwarp][4*subwarp_thx+i] + Alpha[subwarp][subwarp_thx*4 + i]*tempvec1[subwarp][4*subwarp_thx+i+1] + Beta[subwarp][4*subwarp_thx+1]*tempvec1[subwarp][4*subwarp_thx+i+2];

			}

			w = 0.0f;

			for(i = 0; i < 4; i++)
			{
				w += _POW2(tempvec2[subwarp][4*subwarp_thx+1+i]);
			}

			for(k=1; k < dimx_subwarp; k *= 2)
			{
				w += shfl_down64(w, (unsigned int) /* subwarp_null + */ k , 32 );
			}

			w = shfl64(w, subwarp_null , 32);  // this is || tempvec1 ||^2

			w = sqrt(w);
		}
	}

	// compute tridiagonal matrix eigenvalues, determinent in a single thread.  copy determinent and 'Eigenerror' to global memory.


	if(thx == subwarp_null)
	{
		int eigzz;

		for(i =0; i < 4*dimx_subwarp+2; i++)
			Beta[subwarp][i] *= Beta[subwarp][i];

		eigzz = TQLRAT(4*dimx_subwarp, Alpha[subwarp], Beta[subwarp]);



		DetMat[offset ] = DET_TRIDIAG(subn, Alpha[subwarp], Beta[subwarp]);
	//	printf("THX: %d , DET: %g, EIGERR: %d \n", thx, DetMat[offset],eigzz);
		Eigerror[offset] = eigzz;


	}

	// copy the eigenvalues computed above into the global memory.
// i think the error is in here somewhere....
	for(i = 0; i < 4; i++)
	{
		Eigenvalues[i + subwarp_thx*4 + offset * dimx_subwarp * 4] = Alpha[subwarp][i + subwarp_thx*4];

	}

}


/* Estimates supremum of a symmetric real matrix. */
__global__ void UPPER_KER(double *mat, int n, double *Upper)
{
	int i,j,k;
	sms4 smat[32];


	curandState cuState[4];

	clock_t timeseed;

	if(th_null)
	{
		timeseed = clock()+700;
	}

	timeseed = shfl64(timeseed,0,dimx);

	for(i=0; i < 4; i++)
		curand_init((unsigned long long ) timeseed, (unsigned long long) n, (unsigned long long) 4 * thx + i, &cuState[i]);


	for(i=0; i < 4; i++)
	{
		smat[0].vec[i] = curand_normal(&cuState[i]) ; //__sinf((double) (thx*4 + i)); //curand_uniform(dstate1) - 0.5;
	}


	smat[0].b = DOT4(smat[0].vec,smat[0].vec);


	for(k=1; k < dimx; k *= 2)
	{
		smat[0].b += shfl_down64(smat[0].b, k , dimx );
	}


	smat[0].b = shfl64(smat[0].b,0, dimx);
	smat[0].b = sqrt(smat[0].b);	 //__fsqrt_rn(smat[0].b); // intrinsic


	for(i=0; i<4; i++)
	{
		smat[0].vec[i] = smat[0].vec[i] / smat[0].b;
		smat[0].vecw[i] = 0.0;
		smat[0].vec_[i] = 0.0;
	}

	// (II) load 4x4 matrix stractures from main matrix

	for(i=0; i<dimx; i++)
	{
		LOAD_SMS4X(smat[i], mat, 4*dimx, i*4, thx*4);
	}

	for(i=1; i < dimx; i++)
	{
		for(j=0; j < 4; j++)
		{
			smat[i].vec[j] = smat[0].vec[j];
			smat[i].vec_[j] = 0.0;
			smat[i].vecw[j] = 0.0;
		}
	}






	// using l as the index

	int l;

	for(l=0; l < 10; l++)
	{

		for(i=0; i< dimx ; i++)
		{
			EVAL_SMS4(smat[i], smat[0].vec, smat[i].vecw);
		}

		int k;

		for(i=0; i < dimx; i++)
		{


			for(k=1; k < dimx; k *= 2)
			{
				for(j=0; j < 4; j++)
					{ smat[i].vecw[j] += shfl_down64(smat[i].vecw[j], (unsigned int) k , dimx ); }
			}
		}


		for(i=0; i < dimx; i++)
		{
			for(j=0; j < 4; j++)
			{

				smat[i].vecw[j] = shfl64(smat[i].vecw[j], 0, dimx);
			}
		}

		// create next w vector

		smat[0].b = DOT4(smat[0].vecw,smat[0].vecw);

		for(k=1; k <  dimx; k *= 2)
		{
			smat[0].b += shfl_down64(smat[0].b, (unsigned int) k,  dimx);
		}

		smat[0].b = shfl64(smat[0].b,  0, dimx);
		smat[0].b = sqrt(smat[0].b); //__fsqrt_rn(smat[0].b); // intrinsic version


		// set vec_ before setting vec



		for(i=0; i <4 ; i++)
		{
			smat[0].vec[i] = smat[0].vecw[i] / smat[0].b;
		}

		// propagate down , so EVAL works. (unnecessary)
		for(j=1; j < dimx; j++)
		{
			for(i=0; i < 4; i++)
			{
				smat[j].vec[i] = smat[0].vec[i];
			}
		}
	}

	if(th_null)
	{
		*Upper = smat[0].b;

	}

}



__inline__ __device__ void LOAD_SMS4X_INV(sms4 &s4, double Beta[], double theta[], double phi[], int n, int i, int j)
{
	int k, l;

	for(k = 0; k < 4; k++)
		for(l=0; l < 4; l++)
			s4.mat[k*4 + l] = INVERT_TRIDIAG_IJ(n, Beta, theta, phi, (i+k)*n, j + l);

	s4.w = 0;
}

// prints a tridiagonal matrix 
__host__ void print_tridiagonal(double *alpha, double *beta, int n)
{

	int ind1, ind2;
	double val;

	printf(" [ ");


	for(ind1 = 0; ind1 < n; ind1++)
	{

		for(ind2 = 0; ind2 < n; ind2++)
		{
			if(ind1 == ind2)
				val = alpha[ind2];
			else if((ind2 == ind1  - 1 ))
				val = beta[ind2 + 1  ];
			else if ( (ind2 == ind1 + 1 )  )
				val = beta[ind2   ];
			else
				val = 0.0;

			printf(" %g ", val);

			if(ind2 != n - 1)
				printf(", ");

		}
		if(ind1 != n - 1)
			printf("; \n");
		else
			printf("] \n");



	}



}


__host__ void calculate_eigs(double *mat, int n, double *Eigenvalues, double *Det)
{
	dim3 D1(1,1,1);
	dim3 D2((n/4),1,1);

	double *dmat, *dDet, *dEigenvalues; // *dLower
	int *dEigenerror;

	int Eigenerror;

	if( n % 4 != 0 || n > 128)
	{
		fprintf(stderr, " 'int n' must be divisible by 4, and less than or equal to 128! \n");
	}

	cudaDeviceReset();

	cudaMalloc(&dEigenerror, sizeof(int));
	cudaMalloc(&dEigenvalues, sizeof(double)*n);
	cudaMalloc(&dmat, sizeof(double)*n*n);
	cudaMalloc(&dDet, sizeof(double) );
	cudaMemcpy(dmat, mat, sizeof(double)*n*n, cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();

	POS_SYM_MAT_KER<<<D1,D2>>>(dmat, n, dDet, dEigenvalues, dEigenerror);
	cudaDeviceSynchronize();

	cudaMemcpy(Eigenvalues, dEigenvalues, sizeof(double)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(Det, dDet, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Eigenerror, dEigenerror, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	for(int i = 0; i < n; i++)
		printf("%d eigevalue: %g \n", i, Eigenvalues[i]);

}


__host__ void calculate_subeigs(double *mat, int n, double *Eigenvalues, double *Det, int *subIndex, int subn)
{
	dim3 D1(1,1,1);
	dim3 D2((subn/4),1,1);

	double *dmat, *dDet, *dEigenvalues; // *dLower
	int *dEigerror, *dSubIndex;


	if( subn % 4 != 0 || n > 128)
	{
		fprintf(stderr, " 'int n' must be divisible by 4, and less than or equal to 128! \n");
	}

	cudaDeviceReset();

	//cudaMalloc(&dEigenerror, sizeof(int));
	cudaMalloc(&dEigenvalues, sizeof(double)*n);
	cudaMalloc(&dmat, sizeof(double)*n*n);
	cudaMalloc(&dDet, sizeof(double) );
	cudaMalloc(&dSubIndex, sizeof(int)*subn);
	cudaMalloc(&dEigerror, sizeof(int));
	cudaMemcpy(dSubIndex, subIndex, sizeof(int)*subn, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*n*n, cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();
/*
__global__ void POS_SYM_SUBMATRIX_EIGENVALUES_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn) */
	POS_SYM_SUBMATRIX_KER<<<D1,D2>>>(dmat, n, dDet, dEigenvalues, dEigerror, dSubIndex, subn);
	cudaDeviceSynchronize();

	cudaMemcpy(Eigenvalues, dEigenvalues, sizeof(double)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(Det, dDet, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&Eigenerror, dEigenerror, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	for(int i = 0; i < subn; i++)
		printf("%d eigenvalue: %g \n", i, Eigenvalues[i]);

}

__host__ void calculate_subeigs2(double *mat, int n, int matsize, double *Eigenvalues, double *Det, int *subIndex, int subn)
{
	dim3 D1(1,1,1);
	dim3 D2((32),1,1);

	double *dmat, *dDet, *dEigenvalues; // *dLower
	int *dEigerror, *dSubIndex;


	if( subn % 4 != 0 || n > 128)
	{
		fprintf(stderr, " 'int n' must be divisible by 4, and less than or equal to 128! \n");
	}

	printf("welcome to calculate subeigs2...\n");
	cudaDeviceReset();

	//cudaMalloc(&dEigenerror, sizeof(int));
	cudaMalloc(&dEigenvalues, sizeof(double)*n );
	cudaMalloc(&dmat, sizeof(double)*matsize*matsize);
	cudaMalloc(&dDet, sizeof(double) * (32 / subn) );
	cudaMalloc(&dSubIndex, sizeof(int)*n);
	cudaMalloc(&dEigerror, sizeof(int)*(32/subn));
	cudaMemcpy(dSubIndex, subIndex, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*matsize*matsize, cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();
/*
__global__ void POS_SYM_SUBMATRIX_EIGENVALUES_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn) */
	POS_SYM_SUBMATRIX_KER2<<<D1,D2>>>(dmat, matsize, dDet, dEigenvalues, dEigerror, dSubIndex, subn);
	cudaDeviceSynchronize();

	cudaMemcpy(Eigenvalues, dEigenvalues, sizeof(double)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(Det, dDet, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&Eigenerror, dEigenerror, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	for(int i = 0; i < n; i++)
		printf("%d eigenvalue: %g \n", i, Eigenvalues[i]);

}


__host__ void calculate_subeigs3(double *mat, int subIndexLen, int matsize, double *Eigenvalues, double *Det, int *subIndex, int subn)
{

	if( subn % 4 != 0 || subn > 128)
	{
		fprintf(stderr, " 'int subn' must be divisible by 4, and less than or equal to 128! \n");
		return;
	}

	if( subIndexLen % 32 != 0)
	{
		fprintf(stderr, " 'subIndexLen' must be divisible by 32! \n");
		return;
	}

	dim3 D1(subIndexLen / 32,1,1);
	dim3 D2((32),1,1);

	double *dmat, *dDet, *dEigenvalues; // *dLower
	int *dEigerror, *dSubIndex;


	printf("welcome to calculate subeigs2...\n");
	cudaDeviceReset();

	//cudaMalloc(&dEigenerror, sizeof(int));
	cudaMalloc(&dEigenvalues, sizeof(double)*subIndexLen );
	cudaMalloc(&dmat, sizeof(double)*matsize*matsize);
	cudaMalloc(&dDet, sizeof(double) * (subIndexLen / subn) );
	cudaMalloc(&dSubIndex, sizeof(int)*subIndexLen);
	cudaMalloc(&dEigerror, sizeof(int)*(subIndexLen/subn));
	cudaMemcpy(dSubIndex, subIndex, sizeof(int)*subIndexLen, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*matsize*matsize, cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();
/*
__global__ void POS_SYM_SUBMATRIX_EIGENVALUES_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn) */
	POS_SYM_SUBMATRIX_KER2<<<D1,D2>>>(dmat, matsize, dDet, dEigenvalues, dEigerror, dSubIndex, subn);
	cudaDeviceSynchronize();

	cudaMemcpy(Eigenvalues, dEigenvalues, sizeof(double)*subIndexLen, cudaMemcpyDeviceToHost);
	cudaMemcpy(Det, dDet, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&Eigenerror, dEigenerror, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	for(int i = 0; i < subIndexLen; i++)
		printf("%d eigenvalue: %g \n", i, Eigenvalues[i]);

}
