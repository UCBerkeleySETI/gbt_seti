
#include <cuda.h>
#include "guppi2spectra_gpu.h"
#include <stdio.h>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

extern "C" void explode_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);
extern "C" void detect_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern "C" void detectX_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern "C" void detectY_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern "C" void detectV_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern "C" void setQuant(float *lut);
extern "C" void setQuant8(float *lut);
extern "C" void normalize_wrapper(float * tree_dedopplerd_pntr, float *mean, float *stddev, int tdwidth);
extern "C" void vecdivide_wrapper(float * spectrumd, float * divisord, int tdwidth);
extern "C" void explode8_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
extern "C" void explode8init_wrapper(char *channelbufferd, long int length);
extern "C" void explode8simple_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
extern "C" void explode8lut_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);


__constant__ float gpu_qlut[4];
__constant__ float gpu_qlut8[256];
__constant__ float meand;
__constant__ float stddevd;

texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> char_tex;


__global__ void explode8(char *channelbuffer, cufftComplex * voltages, int veclen) {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	 if(tid < veclen) {	 
		  voltages[tid].x = tex1Dfetch(char_tex, 4*tid); 
		  voltages[tid].y = tex1Dfetch(char_tex, 4*tid + 1);
		  voltages[veclen + tid].x = tex1Dfetch(char_tex, 4*tid + 2); 
		  voltages[veclen + tid].y = tex1Dfetch(char_tex, 4*tid + 3);
	 }
	 
}

__global__ void explode8simple(char *channelbuffer, cufftComplex * voltages, int veclen) {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	 if(tid < veclen) {	 
		  voltages[tid].x = (float) channelbuffer[4*tid]; 
		  voltages[tid].y = (float) channelbuffer[4*tid + 1];
		  voltages[veclen + tid].x = (float) channelbuffer[4*tid + 2]; 
		  voltages[veclen + tid].y = (float) channelbuffer[4*tid + 3];
	 }

}

__global__ void explode8lut(unsigned char *channelbuffer, cufftComplex * voltages, int veclen) {

//int tid = threadIdx.x + blockIdx.x * blockDim.x;
//int tid = threadIdx.x + blockDim.x* threadIdx.y +blockDim.x*blockDim.y*blockIdx.x;
//int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	 if(tid < veclen) {	 
		  voltages[tid].x = gpu_qlut8[channelbuffer[4*tid]]; 
		  voltages[tid].y = gpu_qlut8[channelbuffer[4*tid + 1]];
		  voltages[veclen + tid].x = gpu_qlut8[channelbuffer[4*tid + 2]]; 
		  voltages[veclen + tid].y = gpu_qlut8[channelbuffer[4*tid + 3]];
	 }

}

__global__ void explode(unsigned char *channelbuffer, cufftComplex * voltages, int veclen) {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

//float lookup[4];
//lookup[0] = 3.3358750;
//lookup[1] = 1.0;
//lookup[2] = -1.0;
//lookup[3] = -3.3358750;

	 if(tid < veclen) {	 
		  voltages[tid].x = gpu_qlut[(channelbuffer[tid] >> (0 * 2) & 1) +  (2 * (channelbuffer[tid] >> (0 * 2 + 1) & 1))];
		  voltages[tid].y = gpu_qlut[(channelbuffer[tid] >> (1 * 2) & 1) +  (2 * (channelbuffer[tid] >> (1 * 2 + 1) & 1))];
		  voltages[veclen + tid].x = gpu_qlut[(channelbuffer[tid] >> (2 * 2) & 1) +  (2 * (channelbuffer[tid] >> (2 * 2 + 1) & 1))];
		  voltages[veclen + tid].y = gpu_qlut[(channelbuffer[tid] >> (3 * 2) & 1) +  (2 * (channelbuffer[tid] >> (3 * 2 + 1) & 1))];
	 }
	 
}


__global__ void detect(cufftComplex * voltages, int veclen, int fftlen, float * bandpassd, float * spectrumd) {


	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	int indx = tid - (tid%fftlen) + (tid + fftlen/2)%fftlen;


//73 - (73%16) + (73 + 8)%16

	 if(tid < veclen) {
		  //spectrumd[tid] = ((voltages[((tid+fftlen/2)%fftlen)].x * voltages[((tid+fftlen/2)%fftlen)].x) + (voltages[((tid+fftlen/2)%fftlen)].y * voltages[((tid+fftlen/2)%fftlen)].y) + (voltages[fftlen + ((tid+fftlen/2)%fftlen)].x * voltages[fftlen + ((tid+fftlen/2)%fftlen)].x)+ (voltages[fftlen + ((tid+fftlen/2)%fftlen)].y * voltages[fftlen + ((tid+fftlen/2)%fftlen)].y))/bandpassd[tid];	 		  
		  spectrumd[tid] = spectrumd[tid] + ((voltages[indx].x * voltages[indx].x) + (voltages[indx].y * voltages[indx].y) + (voltages[veclen + indx].x * voltages[veclen + indx].x)+ (voltages[veclen + indx].y * voltages[veclen + indx].y));	 		  
	 }
}

/* Detect Stokes V - Uncalibrated - AHHHHHH! */
__global__ void detectV(cufftComplex * voltages, int veclen, int fftlen, float * bandpassd, float * spectrumd) {


	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	int indx = tid - (tid%fftlen) + (tid + fftlen/2)%fftlen;

/* 2 * ( Re(Ey) * Im(Ex) - Re(Ex) * Im(Ey) )  */

	 if(tid < veclen) {
		  spectrumd[tid] = spectrumd[tid] + 2 * ((voltages[veclen + indx].x * voltages[indx].y) - (voltages[indx].x * voltages[veclen + indx].y));	 		  
	 }
}







__global__ void detectX(cufftComplex * voltages, int veclen, int fftlen, float * bandpassd, float * spectrumd) {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;
	
	int indx = tid - (tid%fftlen) + (tid + fftlen/2)%fftlen;

//73 - (73%16) + (73 + 8)%16

	 if(tid < veclen) {
		  //spectrumd[tid] = ((voltages[((tid+fftlen/2)%fftlen)].x * voltages[((tid+fftlen/2)%fftlen)].x) + (voltages[((tid+fftlen/2)%fftlen)].y * voltages[((tid+fftlen/2)%fftlen)].y) + (voltages[fftlen + ((tid+fftlen/2)%fftlen)].x * voltages[fftlen + ((tid+fftlen/2)%fftlen)].x)+ (voltages[fftlen + ((tid+fftlen/2)%fftlen)].y * voltages[fftlen + ((tid+fftlen/2)%fftlen)].y))/bandpassd[tid];	 		  
		  spectrumd[tid] = spectrumd[tid] + ((voltages[indx].x * voltages[indx].x) + (voltages[indx].y * voltages[indx].y));	 		  
	 }
}


__global__ void detectY(cufftComplex * voltages, int veclen, int fftlen, float * bandpassd, float * spectrumd) {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

int indx = tid - (tid%fftlen) + (tid + fftlen/2)%fftlen;

//73 - (73%16) + (73 + 8)%16

	 if(tid < veclen) {
		  //spectrumd[tid] = ((voltages[((tid+fftlen/2)%fftlen)].x * voltages[((tid+fftlen/2)%fftlen)].x) + (voltages[((tid+fftlen/2)%fftlen)].y * voltages[((tid+fftlen/2)%fftlen)].y) + (voltages[fftlen + ((tid+fftlen/2)%fftlen)].x * voltages[fftlen + ((tid+fftlen/2)%fftlen)].x)+ (voltages[fftlen + ((tid+fftlen/2)%fftlen)].y * voltages[fftlen + ((tid+fftlen/2)%fftlen)].y))/bandpassd[tid];	 		  
		  spectrumd[tid] = spectrumd[tid] + ((voltages[veclen + indx].x * voltages[veclen + indx].x)+ (voltages[veclen + indx].y * voltages[veclen + indx].y));	 		  
	 }
}






__global__ void normalize(float * tree_dedopplerd_pntr, int tdwidth)  {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;
	

	 if(tid < tdwidth) { 
		tree_dedopplerd_pntr[tid] = (tree_dedopplerd_pntr[tid] - meand)/stddevd;     
	 }

}

__global__ void vecdivide(float * spectrumd, float * divisord, int tdwidth)  {

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int tid = blockId * blockDim.x + threadIdx.x;

	 if(tid < tdwidth) { 
		spectrumd[tid] = spectrumd[tid]/divisord[tid];     
	 }

}



void explode_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	explode<<<nblocks,nthreads>>>(channelbufferd, voltages, veclen);
}

void detect_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd) {


	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	detect<<<nblocks,nthreads>>>(voltages, veclen, fftlen, bandpassd, spectrumd);	

}

void detectX_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	detectX<<<nblocks,nthreads>>>(voltages, veclen, fftlen, bandpassd, spectrumd);
}

void detectY_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	detectY<<<nblocks,nthreads>>>(voltages, veclen, fftlen, bandpassd, spectrumd);
}

void detectV_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	detectV<<<nblocks,nthreads>>>(voltages, veclen, fftlen, bandpassd, spectrumd);
}

//veclen is number of complex elements, so length of channelbufferd is 4 x veclen
void explode8_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	explode8<<<nblocks,nthreads>>>(channelbufferd, voltages, veclen);
}


//veclen is number of complex elements, so length of channelbufferd is 2 x veclen
void explode8lut_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	explode8lut<<<nblocks,nthreads>>>(channelbufferd, voltages, veclen);
}


void explode8simple_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen) {

	int nthreads = 512;
	dim3 nblocks((veclen+2047)/2048, 4);

	explode8simple<<<nblocks,nthreads>>>(channelbufferd, voltages, veclen);
}

void explode8init_wrapper(char *channelbufferd, long int length) {
	HANDLE_ERROR( cudaBindTexture(0, char_tex, channelbufferd, length) );
}




void setQuant(float *lut) {
#if CUDA_VERSION >= 4500
        fprintf(stderr, "loading lookuptable...%s\n", cudaGetErrorString(cudaMemcpyToSymbol(gpu_qlut, lut, 16, 0, cudaMemcpyHostToDevice)));
#else
        fprintf(stderr, "loading lookuptable...%s\n", cudaGetErrorString(cudaMemcpyToSymbol("gpu_qlut", lut, 16, 0, cudaMemcpyHostToDevice)));
#endif

}

void setQuant8(float *lut) {
#if CUDA_VERSION >= 4500
        fprintf(stderr, "loading lookuptable...%s\n", cudaGetErrorString(cudaMemcpyToSymbol(gpu_qlut8, lut, 1024, 0, cudaMemcpyHostToDevice)));
#else
        fprintf(stderr, "loading lookuptable...%s\n", cudaGetErrorString(cudaMemcpyToSymbol("gpu_qlut8", lut, 1024, 0, cudaMemcpyHostToDevice)));
#endif

}

void normalize_wrapper(float * tree_dedopplerd_pntr, float *mean, float *stddev, int tdwidth) {
	int nthreads = 512;
	dim3 nblocks((tdwidth+2047)/2048, 4);
	cudaMemcpyToSymbol("meand", mean, 4, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("stddevd", stddev, 4, 0, cudaMemcpyHostToDevice);

	normalize<<<nblocks,nthreads>>>(tree_dedopplerd_pntr, tdwidth);
}

void vecdivide_wrapper(float * spectrumd, float * divisord, int tdwidth) {
	int nthreads = 512;
	dim3 nblocks((tdwidth+2047)/2048, 4);
	vecdivide<<<nblocks,nthreads>>>(spectrumd, divisord, tdwidth);
}

