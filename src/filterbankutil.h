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
#include <stdint.h>


void filterbank2fits(char * fitsdata, float *datavec, int nchan, int nsamp, long int hitchan, double snr, double doppler, struct filterbank_input *input);

double filterbank_chan_freq(struct filterbank_input *input, long int channel);

void comp_stats(double *mean, double *stddev, float *vec, long int veclen);
void comp_stats_mad(double *median, double *mad, float *vec, long int veclen);


void normalize (float *vec, long int veclen);

long int sizeof_file(char name[]) ;

long int filterbank_extract_from_buffer(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input);
long int filterbank_extract_from_file(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input);
long int candsearch(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input);
long int candsearch_onoff(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input, struct filterbank_input *offsource);
long int candsearch_doppler(float thresh, struct filterbank_input *input, struct filterbank_input *offsource);
long int candsearch_doppler_mongo(float thresh, struct filterbank_input *input, struct filterbank_input *offsource);


int sum_filterbank(struct filterbank_input *input);

void filterbanksearch_print_usage(); 