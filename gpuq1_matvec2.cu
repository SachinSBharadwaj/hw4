// $ nvcc -arch=sm_61 gpuq1_matvec2.cu -o gpuq1_matvec2 -Xcompiler -fopenmp

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define N 2048
#define M 2048 
#define THREADS_PER_BLOCK 512
#define XBLOCKS (N/THREADS_PER_BLOCK)
#define YBLOCKS M

// MATRIX-VECTOR PRODUCT REFERENCE ON CPU ############################################

void mat_vec( const double* a, const double* b, double* c){ // Refernce product on CPU
  double sum1 = 0.0;
  for (long i = 0; i < M; i++) {
	for(long j = 0; j < N ; j++){
    		sum1 = sum1 + (a[j + i*M] * b[j]);
 	 }
	c[i] = sum1;
	sum1 = 0.0;
   }	

}

// MATRIX-VECTOR PRODUCT ON GPU #####################################################


__global__ void mat_vec_gpu(const double* a, const double* b, double* c){ // Product on GPU
  
  __shared__ double prod[THREADS_PER_BLOCK];
  int blockID  = (gridDim.x*blockIdx.y) + blockIdx.x; 
  int threadID = (blockID * blockDim.x) + threadIdx.x;
  prod[threadIdx.x] = a[threadID] * b[blockIdx.x*THREADS_PER_BLOCK + threadIdx.x]; 

    __syncthreads();

   if(0==threadIdx.x){
  	double S = 0.0;
	for (long i=0; i<THREADS_PER_BLOCK; i++){
		S = S + prod[i];
	}
    
    atomicAdd(&c[blockIdx.y] , S);
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
 
  // ALLOCATING AND INITIALISING THE MATRIX AND VECTORS ON CPU #############################
  double* x = (double*) malloc(M * N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));

  
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    y[i] 	= drand48();
    z[i]	= 0.0;
    z_ref[i]    = 0.0;
    for(long j = 0; j < M; j++){
	x[j+ N*i] = drand48();
    }
  }

  // CPU PRODUCT ###########################################################################

  double tt = omp_get_wtime();
  mat_vec(x, y, z_ref);
  printf("CPU Bandwidth = %f GB/s\n", (2*M*N + M)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

 
  // ALLOCATING AND INITIALISING THE MATRIX AND VECTORS ON GPU #############################

  double *x_d, *y_d, *z_d;
  cudaMalloc((void**)&x_d, M * N *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc((void**)&y_d, N*sizeof(double));
  cudaMalloc((void**)&z_d, M*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, M*sizeof(double), cudaMemcpyHostToDevice);

  // GPU PRODUCT ###########################################################################
  dim3 GridDim(XBLOCKS,YBLOCKS);
  dim3 BlockDim(THREADS_PER_BLOCK); 

  mat_vec_gpu<<< GridDim,BlockDim >>>(x_d, y_d, z_d);

  cudaMemcpy(z, z_d, M* sizeof( double ), cudaMemcpyDeviceToHost);
  printf("GPU Bandwidth = %f GB/s\n", (4*M*N )*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
  // COMPUTING ERROR ########################################################################
  double err = 0;
  err = fabs(z[0]-z_ref[0]);
  printf("Error = %f\n", err);

  // FREEING ALL ALLOCATIONS ################################################################
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);


  return 0;
}

