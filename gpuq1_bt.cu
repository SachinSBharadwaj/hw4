// $ nvcc -arch=sm_61 gpuq1_bt.cu -o gpu03 -Xcompiler -fopenmp

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define N (2048*2048) 		// Vector Size
#define THREADS_PER_BLOCK 512	// Threads in each block

// HOST $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

void vec_dot(double* sum1, const double* a, const double* b){  // Refernce inner product on CPU
  sum1[0] = 0.0;
  for (long i = 0; i < N; i++) {
    sum1[0] = sum1[0] + a[i]*b[i];
    
  }

}

// DEVICE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

__global__ void vec_dot_gpu(double* sum2, const double* a, const double* b){ // Inner product on GPU
  
  __shared__ double DOT[THREADS_PER_BLOCK];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  DOT[threadIdx.x] = a[idx]* b[idx];
   
  __syncthreads();				// To sync all threads upto this stage

  if(0==threadIdx.x){
  	double S = 0.0;
	for (long i=0; i<THREADS_PER_BLOCK; i++){
		S = S + DOT[i];
	} 
	atomicAdd(sum2 , S);			// Atomically add in each thread to prevent race condition.
   }
  
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}



int main() { 

  double* x = (double*) malloc(N * sizeof(double)); // Vector 1
  double* y = (double*) malloc(N * sizeof(double)); // Vector 2
  double* dot = (double*) malloc(sizeof(double));   // Result from GPU
  double* dot_ref = (double*) malloc(sizeof(double)); // Reference result from CPU

  // INITIALISING BOTH THE VECTORS ############################################## 
  	
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
  }

  // REFERENCE INNER PRODUCT ON CPU ############################################

  double tt = omp_get_wtime();
  vec_dot(dot_ref, x, y);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  // INNER PRODUCT ON GPU ######################################################     

  double  *D_d, *x_d, *y_d;  // COPIES OF VECTORS AND RESULT ON GPU DEVICE
  cudaMalloc((void**)&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc((void**)&y_d, N*sizeof(double));
  cudaMalloc((void**)&D_d, sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);

  vec_dot_gpu<<< N/THREADS_PER_BLOCK , THREADS_PER_BLOCK >>>(D_d, x_d, y_d);

  cudaMemcpy(dot, D_d, sizeof( double ), cudaMemcpyDeviceToHost);
  printf("GPU Bandwidth = %f GB/s\n", 4*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  // COMPUTING ERROR BETWEEN CPU AND GPU RESULTS ##############################
  double err = 0;
  err = fabs(dot[0]-dot_ref[0]);
  printf("Error = %f\n", err);

  // FREEING ALL ALLOCATIONS ##################################################
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(D_d);

  free(x);
  free(y);
  free(dot);
  free(dot_ref);


  return 0;
}

