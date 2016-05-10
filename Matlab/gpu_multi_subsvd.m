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
function [eigs, det1, err1] = gpu_multi_subsvd(g, subIndex, minor)

reset(parallel.gpu.GPUDevice.current)

eigs = -1;
det1 = -1;
err1 = -1;

M = size(g,1);
N = size(g, 2);

L = length(subIndex);



if(M ~= N)
    fprintf('dimensions of input for matrix don''t agree! \n');
    return;
end

if(floor(log2(minor)) - log2(minor) ~= 0 )
    fprintf('length of minor is not dyadic!\n');
    return;
end

if(mod(L, minor) ~= 0 )
    fprintf('minor must divide into the length of the index! \n');
    return;
end

if(L > 128)
    fprintf('subindex of length > 128 not supported! \n');
    return;
end

dg = gpuArray(double(g));
ddet = gpuArray(zeros(L/minor,1,'double'));
deigs = gpuArray(zeros(L,1,'double'));
derr = gpuArray(zeros(L/minor,1,'int32'));
dsubIndex = gpuArray(int32(subIndex));

multi_ker = parallel.gpu.CUDAKernel('ParaLanczos.ptx', 'ParaLanczos.cu', 'POS_SYM_SUBMATRIX_KER');

multi_ker.ThreadBlockSize = [minor / 4];
multi_ker.GridSize = [L / minor];


[x, y, z, w, u] = feval(multi_ker, dg, M, ddet, deigs, derr, dsubIndex, L);

eigs = gather(z);
det1 = gather(y);
err1 = gather(w);



end
