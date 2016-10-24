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
#include "filterbankutil.h"


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

long int candsearch(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input);

int sum_filterbank(struct filterbank_input *input);


int main(int argc, char *argv[]) {

	struct filterbank_input sourcea;	
	struct filterbank_input sourceb;
	

	float *diff_spectrum;
	if(argc < 2) {
		exit(1);
	}

	int c;
	long int i,j,k;
	opterr = 0;
 
	while ((c = getopt (argc, argv, "Vvdi:o:c:f:b:s:p:m:a:")) != -1)
	  switch (c)
		{
		case 'a':
		  sourcea.filename = optarg;
		  break;
		case 'b':
		  sourceb.filename = optarg;
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

	
	sourcea.inputfile = fopen(sourcea.filename, "rb");
	read_filterbank_header(&sourcea);
		    
    fprintf(stderr, "Read and summed %d integrations for sourcea\n", sum_filterbank(&sourcea));
	sourceb.inputfile = fopen(sourceb.filename, "rb");

	read_filterbank_header(&sourceb);		    
    fprintf(stderr, "Read and summed %d integrations for sourceb\n", sum_filterbank(&sourceb));

    diff_spectrum = (float*) malloc(sourcea.nchans * sizeof(float));

	//memset(diff_spectrum, 0x0, sourcea.nchans * sizeof(float));


    long int candwidth;
    long int hitchan;
    
    candwidth = 512;
    
    for(i=0;i<sourcea.nchans;i++) diff_spectrum[i] = (sourcea.integrated_spectrum[i] - sourceb.integrated_spectrum[i])/sourceb.integrated_spectrum[i];
	normalize(diff_spectrum, (long int) sourcea.nchans);

	candsearch(diff_spectrum, 512, 10, &sourcea);   


  /* */
/*
	for(j=0;j<164;j++){    
	   for(i = 0; i < 512;i++){
		   fprintf(stderr, "%f,", snap[i + j*512]);
	   }
		 fprintf(stderr, "\n");
    }	
    
    
    for(i=0;i<8;i++) printf("%f, %f, %f\n", sourcea.integrated_spectrum[i], sourceb.integrated_spectrum[i], diff_spectrum[i]); 
*/
	//fprintf(stderr, "src_raj: %lf src_decj: %lf\n", sourcea.src_raj, sourcea.src_dej);
	//fprintf(stderr, "headersize: %d nsamples: %ld datasize: %ld\n", sourcea.headersize, sourcea.nsamples, sourcea.datasize);

	


return 0;

}


long int candsearch(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char fitsname[100];
	float *snap;
	long int startchan;
	long int endchan;
	

	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	fitsdata = (char *) malloc(fitslen);
  	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));

	for(i=0;i<input->nchans;i++) {
	
		if (diff_spectrum[i] > thresh) {
		
			startchan = i - candwidth/2;
			endchan = i + candwidth/2;

			if(endchan > input->nchans) {

				memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
				fprintf(stderr, "%ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, input->nchans, input));
	
			} else if(startchan < 0)    {
	
				memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
				fprintf(stderr, "%ld \n", filterbank_extract_from_file(snap+labs(startchan), 0, input->nsamples, 0, endchan, input));

	
			} else {

				fprintf(stderr, "%ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input));

			}
			
			memset(fitsdata, 0x0, fitslen * sizeof(char));
			filterbank2fits(fitsdata, snap, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, input);
			sprintf(fitsname, "./%s_%5.5f_%ld.fits", input->source_name, input->tstart, i);
			fitsfile = fopen(fitsname, "wb");
			fwrite(fitsdata, 1, fitslen, fitsfile);
			fclose(fitsfile);

		}
	
	}
	free(fitsdata);
	free(snap);
	

}	




int sum_filterbank(struct filterbank_input *input) {
	long int i,j,k;
    input->integrated_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->integrated_spectrum, 0x0, input->nchans * sizeof(float));

    input->temp_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->temp_spectrum, 0x0, input->nchans * sizeof(float));
	j=0;
    while (fread(input->temp_spectrum, sizeof(float), input->nchans, input->inputfile) ) {
           for(i=0;i<input->nchans;i++) input->integrated_spectrum[i] =  input->integrated_spectrum[i] + input->temp_spectrum[i];
    	   j++;
    }
    return j;
}








