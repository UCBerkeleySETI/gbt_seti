#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include "fitsio.h"
#include "filterbank_header.h"
#include "fitshead.h"
#include "imswap.h"

void filterbank2fits(char * fitsdata, float *datavec, int nchan, int nsamp, long int startchan, double snr, double doppler, struct filterbank_input *input);

double filterbank_chan_freq(struct filterbank_input *input, long int channel);

void comp_stats(double *mean, double *stddev, float *vec, long int veclen);

void normalize (float *vec, long int veclen);
