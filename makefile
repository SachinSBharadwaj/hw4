all: gpuq1_bt gpuq1_matvec2 gpuq2_jc


gpuq1_bt: gpuq1_bt.cu
	nvcc -arch=sm_61 gpuq1_bt.cu -o gpuq1_bt -Xcompiler -fopenmp

gpuq1_matvec2: gpuq1_matvec2.cu
	nvcc -arch=sm_61 gpuq1_matvec2.cu -o gpuq1_matvec2 -Xcompiler -fopenmp

gpuq2_jc: gpuq2_jc.cu
	nvcc -arch=sm_61 gpuq2_jc.cu -o gpuq2_jc -Xcompiler -fopenmp
