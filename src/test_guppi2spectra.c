#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "guppi2spectra_gpu.h"

/* Wrapper functions for performing various spectroscopy options on the GPU */
#ifdef __cplusplus
extern "C" {
#endif
extern void explode_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);
extern void detect_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern void setQuant(float *lut);
extern void setQuant8(float *lut);
extern void normalize_wrapper(float * tree_dedopplerd_pntr, float *mean, float *stddev, int tdwidth);
extern void vecdivide_wrapper(float * spectrumd, float * divisord, int tdwidth);
extern void explode8_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
extern void explode8init_wrapper(char *channelbufferd, int length);
extern void explode8simple_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
extern void explode8lut_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
#ifdef __cplusplus
}
#endif

int main(int argc, char *argv[])
{
  // array to store all possible input values
  // This can be thought of as (signed chars):
  //     c0xre, c0xim, c0yre, c0yim, c1xre, ..., c63yim
  // The output of explode8 is (cufftComplex):
  //     (c0xre, c0xim), (c1xre, ..., c63xim), (c0yre, ..., c63yim)
  char ibufh[256];
  cufftComplex obufh[128];
  int i;
  for(i=0; i<256; i++) {
    ibufh[i] = i-128;
  }

  // Device-side storage
  char *ibufd;
  cufftComplex *obufd;

  cudaMalloc((void**)&ibufd, 256 * sizeof(char));
  cudaMalloc((void**)&obufd, 128 * sizeof(cufftComplex));

  // Host to device copy
  cudaMemcpy(ibufd, ibufh, 256*sizeof(char), cudaMemcpyHostToDevice);

  // Explode
  explode8init_wrapper(ibufd, 256);
  explode8_wrapper(ibufd, obufd, 256/4);

  // Device to host copy
  cudaMemcpy(obufh, obufd, 128*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  // Print results
  for(i=0; i<64; i++) {
    printf("%+4d %+4d %+4d %+4d  %+.5f %+.5f %+.5f %+.5f\n",
        ibufh[4*i], ibufh[4*i+1], ibufh[4*i+2],  ibufh[4*i+3],
        obufh[i].x, obufh[i].y,   obufh[i+64].x, obufh[i+64].y);
  }

  return 0;
}
