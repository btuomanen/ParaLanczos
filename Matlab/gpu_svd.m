% This file is part of the ParaLanczos Library 
%
%  Copyright (c) 2015-2016 Brian Tuomanen 
%
%  This library is free software; you can redistribute it and/or
%  modify it under the terms of the GNU Lesser General Public
%  License as published by the Free Software Foundation; either
%  version 3 of the License, or (at your option) any later version.
%  
%  See the file LICENSE included with this distribution for more
%  information. 
%
f
function [eigs, det1, err1] = gpu_svd(g)

reset(parallel.gpu.GPUDevice.current)

eigs = -1;
det1 = -1;
err1 = -1;

M = size(g,1);
N = size(g, 2);

if(M ~= N)
    fprintf('dimensions of input don''t agree! \n');
    return;
end

if(floor(log2(M)) - log2(M) ~= 0 )
    fprintf('dimensions are not dyadic!\n');
    return;
end

if(M > 128)
    fprintf('Matrices larger than 128 x 128 are not supported!\n');
    return;
end

% This is analogous to cudaMalloc followed by cudaMemcpy
dg = gpuArray(g);

% set up zero arrays (analogous to just plain cudaMalloc)
ddet = gpuArray(zeros(1,1,'double'));
deigs = gpuArray(zeros(M,1,'double'));
derr = gpuArray(zeros(1,1,'int32'));

svd_ker = parallel.gpu.CUDAKernel('ParaLanczos.ptx', 'ParaLanczos.cu', 'POS_SYM_MAT_KER');

svd_ker.ThreadBlockSize = [M / 4];

[x, y, z, w] = feval(svd_ker, dg, M, ddet, deigs, derr);

eigs = gather(z);
det1 = gather(y);
err1 = gather(w);

end
