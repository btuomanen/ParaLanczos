/* This file is part of the ParaLanczos Library 

  Copyright (c) 2015-2016 Brian Tuomanen 

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  See the file LICENSE included with this distribution for more
  information. */



#ifndef PARALANCZOS
#define PARALANCZOS







__global__ void DSG_B(double *subgrams, double *Frame, int M, int N, int *subset, int n, int offset1);

__global__ void DSG_B1(double *subgrams, double *Frame, int M, int N, int *subset, int offset1);

__global__ void SG_B1(double *subgrams, double *Frame, int card);

__global__ void POS_SYM_MAT_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror);

__global__ void POS_SYM_SUBMATRIX_KER(double *mat, int n, double *DetMat, double *Eigenvalues, int *Eigerror, int *subIndex, int subn);

__global__ void UPPER_KER(double *mat, int n, double *Upper);


__host__ void print_tridiagonal(double *alpha, double *beta, int n);

__host__ void calculate_eigs(double *mat, int n, double *Eigenvalues, double *Det);

__host__ void calculate_subeigs(double *mat, int n, double *Eigenvalues, double *Det, int *subIndex, int subn);



#endif
