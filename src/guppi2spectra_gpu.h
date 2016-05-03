#include "cufft.h"

__global__ void explode(unsigned char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void detect(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void detectX(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void detectY(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void normalize(float * tree_dedopplerd_pntr, int tdwidth);
__global__ void vecdivide(float * spectrumd, float * divisord, int tdwidth);
__global__ void explode8(char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void explode8simple(char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void explode8lut(unsigned char *channelbuffer, cufftComplex * voltages, int veclen);
