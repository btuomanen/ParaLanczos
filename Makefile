all:
	nvcc -rdc=true -arch=sm_50 -c ParaLanczos.cu -o ParaLanczos.o
	nvcc -rdc=true -arch=sm_50 -ptx ParaLanczos.cu -o ParaLanczos.ptx
	nvcc -rdc=true -arch=sm_50 -c example1.cu -o example1.o
	nvcc -rdc=true -arch=sm_50 ParaLanczos.o example1.o -o example1
	mv ./ParaLanczos.ptx ./Matlab/ParaLanczos.ptx
	cp ./ParaLanczos.cu ./Matlab/ParaLanczos.cu

clean:
	rm example1
	rm *o
	rm ./Matlab/*ptx
