#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include <iostream>
#include <fstream>
//#include "utils.h"
#include <omp.h>
#include <algorithm>
#include <string>
 
#define THREADS_PER_BLOCK 64

// GPU MATRIX VECTOR PRODUCT ON DEVICE ####################################################

__global__ void mat_vec_gpu(const double* a, const double* b, double* c, int f, int N){ // Product on GPU
  
  __shared__ double prod[THREADS_PER_BLOCK];
  int blockID  = (gridDim.x*blockIdx.y) + blockIdx.x; 
  int threadID = (blockID * blockDim.x) + threadIdx.x;
  prod[threadIdx.x] = a[threadID] * b[blockIdx.x*THREADS_PER_BLOCK + threadIdx.x]; 
  if((f == 1) && (threadID%(N+1)==0)){prod[threadIdx.x]=0.0;}

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



int main()
{   printf("DIMENSION     TIME \n");
    for(int n=8;n<=96;n=n+8){
	

	//INITIALISING ALL VARIABLES ***********************************************
	int N 			= n; // Dimension of Matrix
	int N2			= N*N;
	double *f  		= (double *)malloc(N*N*sizeof(double)); // f_i matrix
	double *u1   		= (double *)malloc(N*N*sizeof(double)); // u_i matrix nth time step
	double *u2   		= (double *)malloc(N*N*sizeof(double)); // u_i matrix (n+1)th time step
	double *D   		= (double *)malloc((N2)*(N2)*sizeof(double)); // u_i matrix (n+1)th time step
	double *LHS   		= (double *)malloc(N*N*sizeof(double)); // -1*(delta u_i) matrix
	double *R   		= (double *)malloc(N*N*sizeof(double)); // R_i matrix to store residues
	int REPS    		= 15000;				// Repetitions/Iterations
	double h 		= 1.00/(N+1);				// Discretisation length
	double res,res_init 	= 0.0;					// Residue/Initial Residue
 
	//INITIALISING MATRICES **********************************************
	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(long j=0;j<(N2);j++){	
		f[j]    = -(h*h)*1;
		u1[j]   = 0.0;
		u2[j]   = 0.0;
 		R[j]    = 0.0;
		LHS[j]  = 0.0;
	}
	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(long j=0;j<(N2*N2);j++){	
		D[j]    = 0.0;
		
	}	

	#pragma omp parallel for schedule(static) num_threads(10)
	for(long j=0 ; j<N2 ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			
			if(i==j){

				D[i+j*N2] = 4;
				if(((j+1)%(N))!=0){
					if((i+1)<=N2-1 && (j+1)<=N2-1){
						D[(i+1) + j*N2] = -1;
						D[i + (j+1)*N2] = -1;
					}
				
				}
				if((i+N)<=N2-1 && (j+N)<=N2-1){
					D[(i+N) + j*N2] = -1;
					D[i + (j+N)*N2] = -1;
				}	
			}		
			
		}
		
	}
	
	
	//THE 2D JACOBI ALGORITHM $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

	//t.tic();   
	double t = omp_get_wtime();    			// start timing
	//double sum =0.0;
	int NG = N2;
	int MG = N2;
	int flag = 0;
	int XBLOCKS = (NG/THREADS_PER_BLOCK);
	int YBLOCKS = MG;
	double *z = (double*) malloc(NG*sizeof(double));
	#pragma omp parallel for schedule(static) num_threads(10)
	for (long i= 0; i<NG; i++){z[i]=0.0;}

	double *x_d, *y_d, *z_d;
  	cudaMalloc((void**)&x_d, MG * NG *sizeof(double));
 	Check_CUDA_Error("malloc x failed");
  	cudaMalloc((void**)&y_d, NG*sizeof(double));
  	cudaMalloc((void**)&z_d, MG*sizeof(double));
  	cudaMemcpy(x_d, D, MG*NG*sizeof(double), cudaMemcpyHostToDevice);
  	//cudaMemcpy(y_d, u1, NG*sizeof(double), cudaMemcpyHostToDevice);
  	cudaMemcpy(z_d, z , MG*sizeof(double), cudaMemcpyHostToDevice);
	dim3 GridDim(XBLOCKS,YBLOCKS);
  	dim3 BlockDim(THREADS_PER_BLOCK); 

	
	
	for(long c=1;c<=REPS;c++){ 	// limit of # of iterations

  		
  		cudaMalloc((void**)&y_d, NG*sizeof(double));
  		cudaMalloc((void**)&z_d, MG*sizeof(double));
  		cudaMemcpy(y_d, u1, NG*sizeof(double), cudaMemcpyHostToDevice);
  		cudaMemcpy(z_d, z , MG*sizeof(double), cudaMemcpyHostToDevice);

		flag = 1;
  		mat_vec_gpu<<< GridDim,BlockDim >>>(x_d, y_d, z_d,flag,NG);

  		cudaMemcpy(z, z_d, MG* sizeof( double ), cudaMemcpyDeviceToHost);



	// UPDATING ALL VALUES OF (n+1)th TIME STEP ################################

		#pragma omp parallel for schedule(static) num_threads(10)
		for (long j=0;j<N2 ; j=j+1){
			u2[j] = (1.0 / 4.0) * (f[j] - z[j]);
			u1[j] = u2[j];
			u2[j] = 0.0;
			z[j]  = 0.0;
		
		}
		
		
		cudaFree(y_d);
  		cudaFree(z_d);
		cudaMalloc((void**)&y_d, NG*sizeof(double));
  		cudaMalloc((void**)&z_d, MG*sizeof(double));

		//COMPUTING RESIDUE ################################################

		
  		cudaMemcpy(y_d, u1, NG*sizeof(double), cudaMemcpyHostToDevice);
  		cudaMemcpy(z_d, z , MG*sizeof(double), cudaMemcpyHostToDevice);


  		
		flag = 0;
  		mat_vec_gpu<<< GridDim,BlockDim >>>(x_d, y_d, z_d, flag, NG);

  		cudaMemcpy(z, z_d, MG* sizeof( double ), cudaMemcpyDeviceToHost);
		
		
		#pragma omp parallel for schedule(static) num_threads(10)
		for (long j=0;j<N2 ; j=j+1){ LHS[j]=z[j];  z[j]=0.0;}

		

		#pragma omp parallel for num_threads(10) reduction(+:res)		
		for (long i=0;i<(N2);i++){
			R[i] = LHS[i] - f[i];
	       		res = res + (R[i])*(R[i]);
			LHS[i]=0.0;
			
		
		} 
		res = pow((res),0.5);  


		// CHECKING FOR CONVERGENCE &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		if(c==1){res_init = res;}

		if(c>=1){
		
			if(floor(log10(res_init/res))==6){//printf("Min Error and iter %d \n",c); 
								break;
			}
		}
		res = 0.0;	
		if(c==REPS){printf("End of iterations\n");}

  		cudaFree(y_d);
  		cudaFree(z_d);


	}	cudaFree(x_d);
  		free(z);


		// END OF JACOBI ITERATIONS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

	printf("%d         %f \n",N,omp_get_wtime()-t); //t.toc()); 

	// FREE ALL ALLOCATIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	free(u1);
	free(u2);
	free(f);
	free(LHS);
	free(R);
	free(D);
		

	

 	
  }
	return 0;
}
