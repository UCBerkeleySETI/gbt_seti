#include "cufft.h"

__global__ void explode(unsigned char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void detect(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void detectX(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void detectY(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void detectV(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
__global__ void normalize(float * tree_dedopplerd_pntr, int tdwidth);
__global__ void vecdivide(float * spectrumd, float * divisord, int tdwidth);
__global__ void explode8(char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void explode8simple(char *channelbuffer, cufftComplex * voltages, int veclen);
__global__ void explode8lut(unsigned char *channelbuffer, cufftComplex * voltages, int veclen);

// From $CUDA/samples/common/inc/helper_cuda.h
//
// cuFFT API errors
static const char *_cudaGetCufftErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}
