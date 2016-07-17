all:
	nvcc -rdc=true -arch=sm_50 -c ParaLanczos.cu -o ParaLanczos.o
	nvcc -rdc=true -arch=sm_50 -ptx ParaLanczos.cu -o ParaLanczos.ptx
	nvcc -rdc=true -arch=sm_50 -c example1.cu -o example1.o
	nvcc -rdc=true -arch=sm_50 ParaLanczos.o example1.o -o example1
	nvcc -rdc=true -arch=sm_50 -c example2.cu -o example2.o
	nvcc -rdc=true -arch=sm_50 ParaLanczos.o example2.o -o example2
	nvcc -rdc=true -arch=sm_50 -c example3.cu -o example3.o
	nvcc -rdc=true -arch=sm_50 ParaLanczos.o example3.o -o example3
	mv ./ParaLanczos.ptx ./Matlab/ParaLanczos.ptx
	cp ./ParaLanczos.cu ./Matlab/ParaLanczos.cu

clean:
	rm example1
	rm example2
	rm example3
	rm *o
	rm ./Matlab/*ptx
	rm ./Matlab/*cu
	rm *exe
	rm *exp
	rm *lib
