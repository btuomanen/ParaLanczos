nvcc -arch=sm_50 -c ParaLanczos.cu -o WinParaLanczos.o
nvcc -arch=sm_50 -arch=sm_50 -ptx ParaLanczos.cu -o ParaLanczos.ptx
nvcc -arch=sm_50 -c example1.cu -o wexample1.o
nvcc -arch=sm_50 -c example2.cu -o wexample2.o
nvcc -arch=sm_50 -c example3.cu -o wexample3.o
nvcc -arch=sm_50 wexample1.o WinParaLanczos.o -o WinExample1
nvcc -arch=sm_50 wexample2.o WinParaLanczos.o -o WinExample2
nvcc -arch=sm_50 wexample3.o WinParaLanczos.o -o WinExample3
copy ParaLanczos.cu .\Matlab
copy ParaLanczos.ptx .\Matlab