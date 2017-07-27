/* first pass at on-off/off threshold code for filterbank format data */

#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

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
#include <mysql.h>
#include "setimysql.h"


#define MAXBLOCK 1<<28  //Set maximum size of a memory allocation per file

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



struct taylor_tree {
	long int mlen;
	long int nchn;
 	float *outbuf;
};

struct diff_search {
	int *drift_indexes;
	long int dimX;
	long int dimY;
	float *onsource;
	float *offsource;
	float *result;
	float *maxsnr;
	float *maxdrift;
	float zscore;
	long int nsamples;
};

long int bitrev(long int inval,long int nbits);
void  FlipX(float  Outbuf[], long int xdim, long int ydim);
void taylor_flt(float outbuf[], long int mlen, long int nchn);
void calc_index(float outbuf[], long int dimX, long int dimY, long int nsamples, int *drift_indexes);
void taylor_flt_threaded(void *ptr);
void diff_search_thread(void *ptr);



int main(int argc, char *argv[]) {

	struct filterbank_input sourcea;	
	struct filterbank_input sourceb;
	
	
	struct taylor_tree data[4];

	struct diff_search diffdata[4];
	
	/* enable doppler search mode */
	int dopplersearchmode = 0;

	/* set default values */
	sourcea.zapwidth = 1;
	sourceb.zapwidth = 1;
	

	sourcea.candwidth = 768;
	sourceb.candwidth = 768;

	sourcea.filename = NULL;
	sourceb.filename = NULL;
	
	sourcea.Xpadframes = 4;
	sourceb.Xpadframes = 4;
	
	float zscore = 15;
	float candidatescore=1000;

	sourcea.folder = NULL;
	sourceb.folder = NULL;

	sourcea.diskfolder = NULL;
	sourceb.diskfolder = NULL;

	pthread_t taylor_th0;
	pthread_t taylor_th1;
	pthread_t taylor_th2;
	pthread_t taylor_th3;
	
	pthread_t diff_th0;
	pthread_t diff_th1;
	pthread_t diff_th2;
	pthread_t diff_th3;


	float *diff_spectrum;
	if(argc < 2) {
		exit(1);
	}

	int c;
	long int i,j,k;
	opterr = 0;
 
	while ((c = getopt (argc, argv, "Vvdhf:b:s:a:s:z:d:i:l:w:c:")) != -1)
	  switch (c)
		{
		case 'h':
		  filterbanksearch_print_usage();
		  exit(0);
		  break;
		case 'a':
		  sourcea.filename = strdup(optarg);
		  break;
		case 'i':
		  sourcea.obsid = strdup(optarg);
		  sourceb.obsid = sourcea.obsid;
		  break;
		case 'c':
		  candidatescore = atof(optarg);
          break;
		case 'w':
		  sourcea.candwidth = atoi(optarg);
		  sourceb.candwidth = sourcea.candwidth;
          break;
		case 'l':
		  sourcea.diskfolder = strdup(optarg);
		  sourceb.diskfolder = sourcea.diskfolder;		  
		  break;
		case 'b':
		  sourceb.filename = strdup(optarg);
		  break;
		case 'd':
		  dopplersearchmode = 1;
		  break;
		case 'z':
		  zscore = (float) atof(optarg);
		  break;
		case 's':
		  sourcea.bucketname = strdup(optarg);
		  sourceb.bucketname = sourcea.bucketname;
		  break;
		case 'f':
		  sourcea.folder = strdup(optarg);
		  sourceb.folder = sourcea.folder;
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
		
	if(sourcea.bucketname == NULL || strlen(sourcea.bucketname) < 1) {
		printf("Invalid bucketname: %s  Specify with -s <bucket_name>.\n", sourcea.bucketname);
		exit(1);
	}
	

		
	if(!sourcea.filename || !sourceb.filename) {
		filterbanksearch_print_usage();
		exit(1);
	}
	
	sourcea.inputfile = fopen(sourcea.filename, "rb");
	if(sourcea.inputfile == NULL) {
		fprintf(stderr, "Couldn't open file %s... exiting\n", sourcea.filename);
		exit(-1);
	}
	
	read_filterbank_header(&sourcea);
	
	sourcea.polychannels = (long int) round(fabs(sourcea.nchans * sourcea.foff)/(187.5/64)); 
	sourceb.polychannels = sourcea.polychannels;
	//sourcea.polychannels = 64;
	//sourceb.polychannels = 64;
	

    //fprintf(stderr, "Read and summed %d integrations for sourcea\n", sum_filterbank(&sourcea));

	sourceb.inputfile = fopen(sourceb.filename, "rb");


	if(sourceb.inputfile == NULL) {
		fprintf(stderr, "Couldn't open file %s... exiting\n", sourceb.filename);
		exit(-1);
	}
	

	read_filterbank_header(&sourceb);		    
    //fprintf(stderr, "Read and summed %d integrations for sourceb\n", sum_filterbank(&sourceb));

	long int nsamples;
    if(sourcea.nsamples != sourceb.nsamples) {
    	
    	fprintf(stderr, "ERROR: sample count doesn't match %lf ! (sourcea: %ld, sourceb: %ld)\n", sourcea.fch1, sourcea.nsamples, sourceb.nsamples);
    	nsamples = min(sourcea.nsamples,sourceb.nsamples);
    	sourcea.nsamples = nsamples;
    	sourceb.nsamples = nsamples;
    	fprintf(stderr, "NOW: sample count doesn't match! (sourcea: %ld, sourceb: %ld)\n", sourcea.nsamples, sourceb.nsamples);

    }


	fprintf(stderr, "polyphase channels: %ld\n", sourcea.polychannels);
	fprintf(stderr, "datasize: %ld\n", sourcea.datasize);	    

	sourcea.dimY = (long int) pow(2, ceil(log2(floor(sourcea.nsamples))));
	sourceb.dimY = sourcea.dimY;

	long int channels_to_read=sourcea.polychannels;
	while ( (sourcea.nchans/sourcea.polychannels) * sizeof(float) * channels_to_read * sourcea.dimY > MAXBLOCK) {
		channels_to_read = channels_to_read/2;
	}
		    
	fprintf(stderr, "Will process channels: %ld (%ld bytes) at a time\n", channels_to_read, (sourcea.nchans/sourcea.polychannels) * sizeof(float) * channels_to_read * sourcea.dimY);		    
	
	/* define size of X dimension as total size of data array plus padding on the edges to accommodate edge effects in the Doppler-transform */
	
	
	sourcea.dimX = (sourcea.nchans/sourcea.polychannels) * channels_to_read + (sourcea.Xpadframes * 2 * sourcea.dimY);
	sourceb.dimX = sourcea.dimX;
	
	sourcea.rawdata = (float*) malloc( sizeof(float)  * (sourcea.nchans/sourcea.polychannels) * channels_to_read * sourcea.nsamples );
	sourceb.rawdata = (float*) malloc( sizeof(float)  * (sourcea.nchans/sourcea.polychannels) * channels_to_read * sourcea.nsamples );


	sourcea.data = (float*) malloc( sizeof(float)  * sourcea.dimX * sourcea.dimY);
	sourceb.data = (float*) malloc(sizeof(float) * sourcea.dimX * sourcea.dimY);

	sourcea.datarev = (float*) malloc( sizeof(float)  * sourcea.dimX * sourcea.dimY);
	sourceb.datarev = (float*) malloc(sizeof(float) * sourcea.dimX * sourcea.dimY);

	sourcea.result = (float*) malloc( sizeof(float)  * sourcea.dimX * sourcea.dimY);
	sourceb.result = (float*) malloc(sizeof(float) * sourcea.dimX * sourcea.dimY);
	
	sourcea.revresult = (float*) malloc( sizeof(float)  * sourcea.dimX * sourcea.dimY);
	sourceb.revresult = (float*) malloc(sizeof(float) * sourcea.dimX * sourcea.dimY);

	sourcea.maxsnr = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourcea.maxdrift = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourcea.maxsnrrev = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourcea.maxdriftrev = (float*) malloc( sizeof(float)  * sourcea.dimX);

	sourceb.maxsnr = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourceb.maxdrift = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourceb.maxsnrrev = (float*) malloc( sizeof(float)  * sourcea.dimX);
	sourceb.maxdriftrev = (float*) malloc( sizeof(float)  * sourcea.dimX);




	fprintf(stderr, "Will process channels: %ld (%ld bytes) at a time\n", channels_to_read, (sourcea.nchans/sourcea.polychannels) * sizeof(float) * channels_to_read * sourcea.dimY);		    

	int *drift_indexes;
	drift_indexes = (int *) calloc(sourcea.nsamples, sizeof(int));

	calc_index(sourcea.data, sourcea.dimX, sourcea.dimY, sourcea.nsamples, drift_indexes);

    //diff_spectrum = (float*) malloc(sourcea.dimX * sizeof(float));

	for(i=0;i<sourcea.nsamples;i++) printf("%d \n", drift_indexes[i]);	


		diffdata[0].onsource = sourcea.data;
		diffdata[0].offsource = sourceb.data;
		diffdata[0].result = sourcea.result;
		diffdata[0].maxsnr = sourcea.maxsnr;
		diffdata[0].maxdrift = sourcea.maxdrift;
		
		diffdata[1].onsource = sourceb.data;
		diffdata[1].offsource = sourcea.data;
		diffdata[1].result = sourceb.result;
		diffdata[1].maxsnr = sourceb.maxsnr;
		diffdata[1].maxdrift = sourceb.maxdrift;

		diffdata[2].onsource = sourcea.datarev;
		diffdata[2].offsource = sourceb.datarev;
		diffdata[2].result = sourcea.revresult;
		diffdata[2].maxsnr = sourcea.maxsnrrev;
		diffdata[2].maxdrift = sourcea.maxdriftrev;

		diffdata[3].onsource = sourceb.datarev;
		diffdata[3].offsource = sourcea.datarev;
		diffdata[3].result = sourceb.revresult;
		diffdata[3].maxsnr = sourceb.maxsnrrev;
		diffdata[3].maxdrift = sourceb.maxdriftrev;
		
		for(j=0;j<4;j++) {
			data[j].mlen = sourcea.dimX * sourcea.dimY;
			data[j].nchn = sourcea.dimY;
			diffdata[j].dimX = sourcea.dimX;	
			diffdata[j].dimY = sourcea.dimY;	
			diffdata[j].drift_indexes = drift_indexes;
			diffdata[j].zscore = zscore;
			diffdata[j].nsamples = sourcea.nsamples;
		}

		data[0].outbuf = sourcea.data;		
		data[1].outbuf = sourceb.data;		
		data[2].outbuf = sourcea.datarev;		
		data[3].outbuf = sourceb.datarev;



	float drift_rate_resolution;
	drift_rate_resolution = (sourcea.foff * sourcea.nsamples * sourcea.tsamp); // Hz/sec - guppi chan bandwidth is in MHz



	for(i=0;i<sourcea.polychannels;i=i+channels_to_read){	

		sourcea.currentstartchan = (sourcea.nchans/sourcea.polychannels) * i;
		
		fprintf(stderr,"%ld read!\n",i);
		fflush(stderr);	
		
		filterbank_extract_from_file(sourcea.rawdata, 0, sourcea.nsamples, (sourcea.nchans/sourcea.polychannels) * i,(sourcea.nchans/sourcea.polychannels) * i + (sourcea.nchans/sourcea.polychannels) * channels_to_read , &sourcea);
		filterbank_extract_from_file(sourceb.rawdata, 0, sourcea.nsamples, (sourcea.nchans/sourcea.polychannels) * i,(sourcea.nchans/sourcea.polychannels) * i + (sourcea.nchans/sourcea.polychannels) * channels_to_read , &sourceb);	
		
		memset(sourcea.data, 0x0, sizeof(float) * sourcea.dimX * sourcea.dimY);
		memset(sourceb.data, 0x0, sizeof(float) * sourcea.dimX * sourcea.dimY);
	
		for(j=0;j<sourcea.nsamples;j++) {
			memcpy(sourcea.data + (j*sourcea.dimX) + (sourcea.dimY*sourcea.Xpadframes), sourcea.rawdata + (j * (sourcea.nchans/sourcea.polychannels) * channels_to_read), (sourcea.nchans/sourcea.polychannels) * channels_to_read * sizeof(float));
			memcpy(sourceb.data + (j*sourcea.dimX) + (sourcea.dimY*sourcea.Xpadframes), sourceb.rawdata + (j * (sourcea.nchans/sourceb.polychannels) * channels_to_read), (sourcea.nchans/sourcea.polychannels) * channels_to_read * sizeof(float));
		}
		
		
		
		fprintf(stderr,"%ld read!\n",i);
		fflush(stderr);	
		
		memcpy(sourcea.datarev, sourcea.data, sourcea.dimX * sourcea.dimY * sizeof(float));
		memcpy(sourceb.datarev, sourceb.data, sourceb.dimX * sourceb.dimY * sizeof(float));

		FlipX(sourcea.datarev, sourcea.dimX, sourcea.dimY);
		FlipX(sourceb.datarev, sourceb.dimX, sourceb.dimY);

				
		
		//for(k=0;k<10;k++) printf("%g %g %g \n", sourcea.data[i], sourceb.data[i],result[i]);


		pthread_create (&taylor_th0, NULL, (void *) &taylor_flt_threaded, (void *) &data[0]);
		pthread_create (&taylor_th1, NULL, (void *) &taylor_flt_threaded, (void *) &data[1]);
		pthread_create (&taylor_th2, NULL, (void *) &taylor_flt_threaded, (void *) &data[2]);
		pthread_create (&taylor_th3, NULL, (void *) &taylor_flt_threaded, (void *) &data[3]);

		//taylor_flt(sourcea.data, sourcea.dimX * sourcea.dimY, sourcea.dimY);
		//taylor_flt(sourceb.data, sourcea.dimX * sourcea.dimY, sourcea.dimY);
		//taylor_flt(sourcea.datarev, sourcea.dimX * sourcea.dimY, sourcea.dimY);
		//taylor_flt(sourceb.datarev, sourcea.dimX * sourcea.dimY, sourcea.dimY);

		pthread_join(taylor_th0, NULL);
		pthread_join(taylor_th1, NULL);
		pthread_join(taylor_th2, NULL);
		pthread_join(taylor_th3, NULL);
		
		fprintf(stderr,"%ld doppler!\n",i);	


//		for(j=0;j<;j++) {
//			memcpy(sourcea.data + (j*(sourcea.dimY*sourcea.Xpadframes*2 + sourcea.dimX)) + (sourcea.dimY*sourcea.Xpadframes), sourcea.rawdata + (j * (sourcea.nchans/sourcea.polychannels) * channels_to_read), (sourcea.nchans/sourcea.polychannels) * channels_to_read * sizeof(float));
//			memcpy(sourceb.data + (j*(sourcea.dimY*sourcea.Xpadframes*2 + sourcea.dimX)) + (sourcea.dimY*sourcea.Xpadframes), sourceb.rawdata + (j * (sourcea.nchans/sourceb.polychannels) * channels_to_read), (sourcea.nchans/sourcea.polychannels) * channels_to_read * sizeof(float));
//		}


/*	

		for(j=0;j<sourcea.nsamples;j++) {
        		for(k=0;k<sourcea.dimX;k++) diff_spectrum[k] = (sourcea.data[k+(drift_indexes[j] * sourcea.dimX)] - sourceb.data[k+(drift_indexes[j] * sourcea.dimX)])/sourceb.data[k+(drift_indexes[j] * sourcea.dimX)];
		}

		for(j=0;j<sourcea.nsamples;j++) {
        		for(k=0;k<sourcea.dimX;k++) diff_spectrum[k] = (sourceb.data[k+(drift_indexes[j] * sourceb.dimX)] - sourcea.data[k+(drift_indexes[j] * sourcea.dimX)])/sourcea.data[k+(drift_indexes[j] * sourcea.dimX)];
		}

		for(j=0;j<sourcea.nsamples;j++) {
        		for(k=0;k<sourcea.dimX;k++) diff_spectrum[k] = (sourcea.datarev[k+(drift_indexes[j] * sourcea.dimX)] - sourceb.datarev[k+(drift_indexes[j] * sourcea.dimX)])/sourceb.datarev[k+(drift_indexes[j] * sourcea.dimX)];
		}

		for(j=0;j<sourcea.nsamples;j++) {
        		for(k=0;k<sourcea.dimX;k++) diff_spectrum[k] = (sourceb.datarev[k+(drift_indexes[j] * sourceb.dimX)] - sourcea.datarev[k+(drift_indexes[j] * sourcea.dimX)])/sourcea.datarev[k+(drift_indexes[j] * sourcea.dimX)];
		}
*/


		
		
		memset(sourcea.maxsnr, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourcea.maxdrift, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourcea.maxsnrrev, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourcea.maxdriftrev, 0x0, sizeof(float) * sourcea.dimX);

		memset(sourceb.maxsnr, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourceb.maxdrift, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourceb.maxsnrrev, 0x0, sizeof(float) * sourcea.dimX);
		memset(sourceb.maxdriftrev, 0x0, sizeof(float) * sourcea.dimX);

		fprintf(stderr,"%ld launch diff!\n",i);	
		fflush(stderr);

		pthread_create (&diff_th0, NULL, (void *) &diff_search_thread, (void *) &diffdata[0]);
		pthread_create (&diff_th1, NULL, (void *) &diff_search_thread, (void *) &diffdata[1]);
		pthread_create (&diff_th2, NULL, (void *) &diff_search_thread, (void *) &diffdata[2]);
		pthread_create (&diff_th3, NULL, (void *) &diff_search_thread, (void *) &diffdata[3]);

		pthread_join(diff_th0, NULL);
		pthread_join(diff_th1, NULL);
		pthread_join(diff_th2, NULL);
		pthread_join(diff_th3, NULL);

		FlipX(sourcea.maxsnrrev, sourcea.dimX, 1);
		FlipX(sourcea.maxdriftrev, sourcea.dimX, 1);
		FlipX(sourceb.maxsnrrev, sourcea.dimX, 1);
		FlipX(sourceb.maxdriftrev, sourcea.dimX, 1);
		
		for(j=(sourcea.Xpadframes * sourcea.dimY);j<(sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));j++) {
			
			if (sourcea.maxsnr[j] > 0) {

				/* higher SNR hit at a nearby frequency - zero out candidate if there is a higher SNR hit within +/- candwidth */
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(j != k && sourcea.maxsnr[k] > sourcea.maxsnr[j]) sourcea.maxsnr[j] = 0;
				}
			
				/* higher SNR hit in the neg drift search at a nearby frequency - same as above, but search the negative drift rates */
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourcea.maxsnrrev[k] > sourcea.maxsnr[j]) sourcea.maxsnr[j] = 0;
				}

				/* hit in the off-source at a nearby frequency */
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourceb.maxsnr[k] > 0) sourcea.maxsnr[j] = 0;
				}								

				/* hit in the off-source neg drift at a nearby frequency*/
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourceb.maxsnrrev[k] > 0) sourcea.maxsnr[j] = 0;
				}								
	
				/* same candidate detected in drift rate = 0 and -0 */
				if(fabsf(sourcea.maxsnrrev[j] - sourcea.maxsnr[j]) < 0.001) sourcea.maxsnr[j] = 0;

				/* high pass filter method moves candidates by 1 bin */
				if(fabsf(sourcea.maxsnrrev[j+1] - sourcea.maxsnr[j]) < 0.001) sourcea.maxsnr[j] = 0;
							
			}
			
			if (sourcea.maxsnrrev[j] > 0) {

				/* higher SNR hit at a nearby time */
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(j != k && sourcea.maxsnrrev[k] > sourcea.maxsnrrev[j]) sourcea.maxsnrrev[j] = 0;
				}
			
				/* higher SNR hit in the pos drift search at a nearby time */
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourcea.maxsnr[k] > sourcea.maxsnrrev[j]) sourcea.maxsnrrev[j] = 0;
				}

				/* hit in the off-source at a nearby time*/
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourceb.maxsnrrev[k] > 0) sourcea.maxsnrrev[j] = 0;
				}								

				/* hit in the off-source neg drift at a nearby time*/
				for(k = max(0, j - (sourcea.candwidth/2)); k < min(j + (sourcea.candwidth/2), sourcea.dimX - (sourcea.Xpadframes * sourcea.dimY));k++) {
				   if(sourceb.maxsnr[k] > 0) sourcea.maxsnrrev[j] = 0;
				}								

			
			}
			
			
		}


		candsearch_doppler_mongo(candidatescore, &sourcea, &sourceb);
		//candsearch_doppler(20, &sourcea, &sourceb);


		//for(j=(sourcea.Xpadframes * sourcea.dimY);j<(sourcea.dimX - (2 * sourcea.Xpadframes * sourcea.dimY));j++) {
		//	if (sourcea.maxsnr[j] > 0) printf("Cand at %ld of %ld: onsource snr: %g drift: %g Hz/sec off source snr: %g\n", j, sourcea.dimX, sourcea.maxsnr[j], sourcea.maxdrift[j] * drift_rate_resolution, sourceb.maxsnr[j]);
		//}




	}
	




/*
		for(j=0;j<sourcea.dimY;j++) foo = foo + sourcea.data[(sourcea.nchans/sourcea.polychannels) * channels_to_read * j];		
		for(j=0;j<sourcea.dimY;j++) foo2 = foo2 + sourcea.data[(sourcea.nchans/sourcea.polychannels) * channels_to_read * j + 1];		
		for(j=0;j<sourcea.dimY;j++) foo3 = foo3 + sourcea.data[(sourcea.nchans/sourcea.polychannels) * channels_to_read * j + 2];		
			
		fprintf(stderr,"%ld read %f %f!\n",i, sourcea.integrated_spectrum[(sourcea.nchans/sourcea.polychannels) * i], foo);	
		fprintf(stderr,"%ld read %f %f!\n",(sourcea.nchans/sourcea.polychannels) * i, sourcea.integrated_spectrum[(sourcea.nchans/sourcea.polychannels) * i+1], foo2);	
		fprintf(stderr,"%ld read %f %f!\n",i, sourcea.integrated_spectrum[(sourcea.nchans/sourcea.polychannels) * i+2], foo3);	
*/	


	exit(0);	    




	//memset(diff_spectrum, 0x0, sourcea.nchans * sizeof(float));


/*
for(i = 0; i < sourcea.nchans; i++) {
	    if( (i + (sourcea.nchans/sourcea.polychannels)/2)%(sourcea.nchans/sourcea.polychannels) == 0 ) {
			    			  
			  left =  i - (long int) ceil((sourcea.zapwidth + 1) / 2);
			  right = i + (long int) floor((sourcea.zapwidth + 1) / 2);
			  //fprintf(stderr, "ZAPPING\n", left, right);
			  

				   if(left >= 0 && right < sourcea.nchans) {
						 mean = (diff_spectrum[left] + diff_spectrum[right])/2;
				   } else if (left < 0 && right < sourcea.nchans) {
						 mean = (diff_spectrum[right]);
				   } else if (left >= 0 && right >= sourcea.nchans) {
						 mean = (diff_spectrum[left]);
				   }

				   for(k = max(0, (left+1));k < min(right, sourcea.nchans);k++) {
						diff_spectrum[k] = mean;				   
					}			  
			  }

	}					  
			  
*/	


	
	
	


/* array will need to be grown */

/* time steps must be a power of 2 and edges will need padding */
/* padding on edges should be n_timesteps larger */

long int gulplength = sourcea.nsamples*sourcea.nchans;



/*


sourcea.dimY = (long int) pow(2, ceil(log2(floor(sourcea.nsamples))));
fprintf(stderr, "new total time steps in dedoppler matrix: %ld\n", sourcea.dimY); 
sourcea.dimX = sourcea.nchans + (8 * sourcea.dimY);


if(padleft == NULL && padright == NULL) {
	padleft = (float*) malloc(4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	padright = (float*) malloc(4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	memset(padleft, 0x0, 4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	memset(padright, 0x0, 4 * sourcea.dimY * sourcea.dimY * sizeof(float));
} 


*/
    





//loop over number of gulps

//gulp data

//zero left and right padding


fprintf(stderr, "settled on gulp length: %ld\n", gulplength);

sourcea.data = (float*) malloc(gulplength * sizeof(float));
sourceb.data = (float*) malloc(gulplength * sizeof(float));



/* malloc arrays for padding on the left and padding on the right */
	


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



/*  ======================================================================  */
/*  This is a function to Taylor-tree-sum a data stream. It assumes that    */
/*  the arrangement of data stream is, all points in first spectra, all     */
/*  points in second spectra, etc...  Data are summed across time           */
/*                     Original version: R. Ramachandran, 07-Nov-97, nfra.  */
/*                     Modified 2011 A. Siemion float/64 bit addressing     */
/*  outbuf[]       : input array (float), replaced by dedispersed data  */
/*                   at the output                                          */
/*  mlen           : dimension of outbuf[] (long int)                            */
/*  nchn           : number of frequency channels (long int)                     */
/*                                                                          */
/*  ======================================================================  */

void taylor_flt(float outbuf[], long int mlen, long int nchn)
{
  float itemp;
  long int   nsamp,npts,ndat1,nstages,nmem,nmem2,nsec1,nfin, i;
  long int   istages,isec,ipair,ioff1,i1,i2,koff,ndelay,ndelay2;
  long int   bitrev(long int, long int);

  /*  ======================================================================  */

  nsamp   = ((mlen/nchn) - (2*nchn));
  npts    = (nsamp + nchn);
  ndat1   = (nsamp + 2 * nchn);
  
  //nstages = (int)(log((float)nchn) / 0.6931471 + 0.5);
  nstages = (long int) log2((double)nchn);
  nmem    = 1;


  for (istages=0; istages<nstages; istages++) {
    nmem  *= 2;
    nsec1  = (nchn/nmem);
    nmem2  = (nmem - 2);

    for (isec=0; isec<nsec1; isec++) {
      ndelay = -1;
      koff   = (isec * nmem);

      for (ipair=0; ipair<(nmem2+1); ipair += 2) {
        

        ioff1   = (bitrev(ipair,istages+1)+koff) * ndat1;
        i2      = (bitrev(ipair+1,istages+1) + koff) * ndat1;
        ndelay++;
        ndelay2 = (ndelay + 1);
        nfin    = (npts + ioff1);
        for (i1=ioff1; i1<nfin; i1++) {

          itemp      = (outbuf[i1] + outbuf[i2+ndelay]);
          outbuf[i2] = (outbuf[i1] + outbuf[i2+ndelay2]);
          outbuf[i1] = itemp;
          i2++;

        }
      }
    }
  }

  return;
}


void diff_search_thread(void *ptr) {

	long int i,j,k;
	
	long int dimX;
	long int dimY;
	float *onsource;
	float *offsource;
	float *result;
	float *maxsnr;
	float *maxdrift;
	int * drift_indexes;
	int indx;
	float zscore;
	long int nsamples;
	
	struct diff_search *diffdata;
	diffdata = (struct diff_search *) ptr;

	dimX = diffdata->dimX;
	dimY = diffdata->dimY;
	onsource = diffdata->onsource;
	offsource = diffdata->offsource;
	result = diffdata->result;
	maxsnr = diffdata->maxsnr;
	maxdrift = diffdata->maxdrift;
	zscore = diffdata->zscore;
	nsamples = diffdata->nsamples;

	drift_indexes = diffdata->drift_indexes;

/*
	for(i=0;i<(dimX*dimY);i++){

		// on - off / off
		//result[i] = (onsource[i] - offsource[i]) / offsource[i];


		//result[i] = onsource[i];
		
		if(isnan(result[i]) || isinf(result[i])) result[i] = 0;
	}
*/	
	


	/* high pass filter */
	for(i=0;i<(dimX*dimY);i = i + dimX){
		for(j=i;j<dimX + i - 1;j++) result[j] = fabsf(onsource[j] - onsource[j+1]);		
		result[dimX + i - 1] = 0;
	}
	
	//for(i=1000;i<1010;i++) printf("before %g %g %g \n", onsource[i], offsource[i],result[i]);

	for(i=0;i<(dimX*dimY);i = i + dimX) normalize(result + i, dimX);

	//for(i=1000;i<1010;i++) printf("after %g %g %g \n", onsource[i], offsource[i],result[i]);
	
	/* scan through all hits and note the snr and drift rate of the most significant */

	for(i=0;i<(nsamples);i = i + 1) {
		indx = drift_indexes[i];
		//printf("running zscore: %g index: %d...\n", zscore, indx);
		for(j=0;j<dimX;j++){
	     	if (result[indx * dimX + j] > zscore && result[indx * dimX + j] > maxsnr[j]) {		
	     		maxsnr[j] = result[indx * dimX + j];
	     		maxdrift[j] = (float) i;
	     	}
		}	
	}
	
	
	
	

	return;
}



void taylor_flt_threaded(void *ptr)
{

   float *outbuf; 
   long int mlen; 
   long int nchn;

   struct taylor_tree *data;
   data = (struct taylor_tree *) ptr;

   outbuf = data->outbuf;
   mlen = data->mlen;
   nchn = data->nchn;

  float itemp;
  long int   nsamp,npts,ndat1,nstages,nmem,nmem2,nsec1,nfin, i;
  long int   istages,isec,ipair,ioff1,i1,i2,koff,ndelay,ndelay2;
  long int   bitrev(long int, long int);

  /*  ======================================================================  */

  nsamp   = ((mlen/nchn) - (2*nchn));
  npts    = (nsamp + nchn);
  ndat1   = (nsamp + 2 * nchn);
  
  //nstages = (int)(log((float)nchn) / 0.6931471 + 0.5);
  nstages = (long int) log2((double)nchn);
  nmem    = 1;


  for (istages=0; istages<nstages; istages++) {
    nmem  *= 2;
    nsec1  = (nchn/nmem);
    nmem2  = (nmem - 2);

    for (isec=0; isec<nsec1; isec++) {
      ndelay = -1;
      koff   = (isec * nmem);

      for (ipair=0; ipair<(nmem2+1); ipair += 2) {
        

        ioff1   = (bitrev(ipair,istages+1)+koff) * ndat1;
        i2      = (bitrev(ipair+1,istages+1) + koff) * ndat1;
        ndelay++;
        ndelay2 = (ndelay + 1);
        nfin    = (npts + ioff1);
        for (i1=ioff1; i1<nfin; i1++) {

          itemp      = (outbuf[i1] + outbuf[i2+ndelay]);
          outbuf[i2] = (outbuf[i1] + outbuf[i2+ndelay2]);
          outbuf[i1] = itemp;
          i2++;

        }
      }
    }
  }

  return;
}




/*  ======================================================================  */
/*  This function bit-reverses the given value "inval" with the number of   */
/*  bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.               */
/*  ======================================================================  */

long int bitrev(long int inval,long int nbits)
{
     long int     ifact,k,i,ibitr;

     if(nbits <= 1)
     {
          ibitr = inval;
     }
     else
     {
          ifact = 1;
          for (i=1; i<(nbits); ++i)
               ifact  *= 2;
          k     = inval;
          ibitr = (1 & k) * ifact;

          for (i=2; i < (nbits+1); i++)
          {
               k     /= 2;
               ifact /= 2;
               ibitr += (1 & k) * ifact;
          }
     }
     return ibitr;
}


void calc_index(float outbuf[], long int dimX, long int dimY, long int nsamples, int *drift_indexes) {

	   long int indx;
	   long int i,j,k;

	   long int *ibrev;
	   memset(outbuf, 0x0, sizeof(float) * dimX * dimY);

	   /* build index mask for in-place tree doppler correction */	
	   ibrev = (long int *) calloc(dimY, sizeof(long int));

	   for (i=0; i<dimY; i++) {
		   ibrev[i] = bitrev((long int) i, (long int) log2((double) dimY));
		   //printf("nsamples: %ld dimY: %ld dimX: %ld offset: %ld\n", nsamples, dimY, dimX, ibrev[i]);
	   }

	   /* solve for the indices of unique doppler drift rates */
	   /* if we pad with zero time steps, we'll double up on a few drift rates as we step through */
	   /* all 2^n steps */
	   /* place a counter into dedoppler array in the last valid spectrum of the array */
	   /* perform a dedoppler correction, then identify the indices corresponding to unique values of the counter */

	   for(i=0;i<dimY;i++) {
		   outbuf[dimX * (nsamples - 1) + i] = (float) i; 
	   }

	   taylor_flt(outbuf, dimX * dimY, dimY);

	   k = -1;
	   for(i=0;i<dimY;i++){
		  indx  = (ibrev[i] *  dimX);
			   if(outbuf[indx] != k) {
				   k = outbuf[indx];
				   drift_indexes[k]=ibrev[i];
				   //printf("time index: %02ld Sum: %02f\n", i, outbuf[indx]);				
			   }	
	   }

	return;
}




void  FlipX(float  Outbuf[], 
               long int    xdim, 
               long int    ydim) 
               {

  long int    indx, i, j, revi;
  float *temp;

  temp  = (float *) calloc((xdim), sizeof(float));
	
  	
 
   for(j = 0; j < ydim; j++) {
		 revi = xdim - 1;

		 indx = j * xdim;

		 memcpy(temp, Outbuf + (indx), (sizeof(float)*xdim));

		 for(i = 0; i < xdim;i++) {
			Outbuf[(indx) + i] = temp[revi];
			revi--;
		 }
	}  
  
  free(temp);

  return;
}





