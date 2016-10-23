#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "fitsio.h"
#include "filterbank_header.h"
#include "fitshead.h"
#include "imswap.h"

void filterbank2fits(char * fitsdata, float *datavec, int nchan, int nsamp, long int startchan, double snr, double doppler, struct filterbank_input *input);

double filterbank_chan_freq(struct filterbank_input *input, long int channel);

void comp_stats(double *mean, double *stddev, float *vec, long int veclen);

void normalize (float *vec, long int veclen);

long long sizeof_file(char name[]) ;

long int filterbank_extract_from_file(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input);
