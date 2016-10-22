/* first pass at on-off/off threshold code for filterbank format data */


/*Basic plan:  */
/* on command line, take two files as input, on source and off source   */
/* read and integrate both observations */
/* compute on-off/off and off-on/on in two temporary buffers  */
/* normalize both by subtracting mean and dividing through by std dev (mean = 0, stddev = 1) */
/* determine location of hits */
/* produce fits output for all hits */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include "fitsio.h"
#include "psrfits.h"
#include "guppi_params.h"
#include "fitshead.h"
#include "median.h"
#include "setimysql.h"
#include <fftw3.h>
#include <sys/stat.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_multifit.h>
#include "barycenter.h"
#include <pthread.h>
#include "filterbank_header.h"


/*
Data file                        : blc07_guppi_57601_39735_Hip116819_0061.gpuspec.0002.fil
Header size (bytes)              : 384
Data size (bytes)                : 15204352
Data type                        : filterbank (topocentric)
Telescope                        : GBT
Datataking Machine               : ?????
Source Name                      : Hip116819
Source RA (J2000)                : 23:40:38.1
Source DEC (J2000)               : -18:59:19.7
Frequency of channel 1 (MHz)     : 938.963413
Channel bandwidth      (MHz)     : -0.002861
Number of channels               : 65536
Number of beams                  : 1
Beam number                      : 1
Time stamp of first sample (MJD) : 57601.459895833330
Gregorian date (YYYY/MM/DD)      : 2016/08/01
Sample time (us)                 : 1073741.82400
Number of samples                : 58
Observation length (minutes)     : 1.0
Number of bits per sample        : 32
Number of IFs                    : 1

*/

int sum_filterbank(struct filterbank_input *input);


int main(int argc, char *argv[]) {

	struct filterbank_input sourcea;	
	struct filterbank_input sourceb;
	
	float *diff_spectrum;
	if(argc < 2) {
		exit(1);
	}

	int c;
	int i,j,k;
	opterr = 0;
 
	while ((c = getopt (argc, argv, "Vvdi:o:c:f:b:s:p:m:a:")) != -1)
	  switch (c)
		{
		case 'a':
		  sourcea.rawdatafile = optarg;
		  break;
		case 'b':
		  sourceb.rawdatafile = optarg;
		  break;
		case '?':
		  if (optopt == 'i' || optopt == 'o' || optopt == '1' || optopt == '2' || optopt == '3' || optopt == '4' || optopt == '5' || optopt == '6'|| optopt == '7' || optopt == '8')
			fprintf (stderr, "Option -%c requires an argument.\n", optopt);
		  else if (isprint (optopt))
			fprintf (stderr, "Unknown option `-%c'.\n", optopt);
		  else
			fprintf (stderr,
					 "Unknown option character `\\x%x'.\n",
					 optopt);
		  return 1;
		default:
		  abort ();
		}

	
	sourcea.inputfile = fopen(sourcea.rawdatafile, "rb");
	read_filterbank_header(&sourcea);
		    
    printf("Read and summed %d integrations for sourcea\n", sum_filterbank(&sourcea));
	sourceb.inputfile = fopen(sourceb.rawdatafile, "rb");

	read_filterbank_header(&sourceb);		    
    printf("Read and summed %d integrations for sourceb\n", sum_filterbank(&sourceb));

    diff_spectrum = (float*) malloc(sourcea.nchans * sizeof(float));
	memset(diff_spectrum, 0x0, sourcea.nchans * sizeof(float));

    for(i=0;i<sourcea.nchans;i++) diff_spectrum[i] = (sourcea.integrated_spectrum[i] - sourceb.integrated_spectrum[i])/sourceb.integrated_spectrum[i];

    for(i=0;i<sourcea.nchans;i++) printf("%f, %f, %f\n", sourcea.integrated_spectrum[i], sourceb.integrated_spectrum[i], diff_spectrum[i]); 

	//fprintf(stderr, "src_raj: %lf src_decj: %lf\n", sourcea.src_raj, sourcea.src_dej);


return 0;

}


int sum_filterbank(struct filterbank_input *input) {
	int i,j,k;
    input->integrated_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->integrated_spectrum, 0x0, input->nchans * sizeof(float));

    input->temp_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->temp_spectrum, 0x0, input->nchans * sizeof(float));
	j=0;
    while (fread(input->temp_spectrum, sizeof(float), input->nchans, input->inputfile)) {
           for(i=0;i<input->nchans;i++) input->integrated_spectrum[i] =  input->integrated_spectrum[i] + input->temp_spectrum[i];
    	   j++;
    }
    return j;
}
