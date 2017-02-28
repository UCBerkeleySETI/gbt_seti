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

#DEFINE MAXBLOCK 1<<31  //Set maximum size of a memory allocation per file

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




int main(int argc, char *argv[]) {

	struct filterbank_input sourcea;	
	struct filterbank_input sourceb;
	
	/* enable doppler search mode */
	int dopplersearchmode = 0;

	/* set default values */
	sourcea.zapwidth = 1;
	sourceb.zapwidth = 1;
	
	sourcea.polychannels = 64;
	sourceb.polychannels = 64;

	sourcea.candwidth = 512;
	sourceb.candwidth = 512;


	float *diff_spectrum;
	if(argc < 2) {
		exit(1);
	}

	int c;
	long int i,j,k;
	opterr = 0;
 
	while ((c = getopt (argc, argv, "Vvdi:b:p:a:w:")) != -1)
	  switch (c)
		{
		case 'a':
		  sourcea.filename = optarg;
		  break;
		case 'b':
		  sourceb.filename = optarg;
		  break;
		case 'p':
		  sourcea.polychannels = atoi(optarg);
		  sourceb.polychannels = sourcea.polychannels;
		  break;
		case 'w':
		  sourcea.zapwidth = atoi(optarg);
		  sourceb.zapwidth = sourcea.zapwidth;
		  break;

		case 'd':
		  dopplersearchmode = 1;
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
	if(sourcea.inputfile == NULL) {
		fprintf(stderr, "Couldn't open file %s... exiting\n", sourcea.filename);
		exit(-1);
	}
	
	read_filterbank_header(&sourcea);
		    
    fprintf(stderr, "Read and summed %d integrations for sourcea\n", sum_filterbank(&sourcea));

	sourceb.inputfile = fopen(sourceb.filename, "rb");

	if(sourceb.inputfile == NULL) {
		fprintf(stderr, "Couldn't open file %s... exiting\n", sourceb.filename);
		exit(-1);
	}
	

	read_filterbank_header(&sourceb);		    
    fprintf(stderr, "Read and summed %d integrations for sourceb\n", sum_filterbank(&sourceb));

    diff_spectrum = (float*) malloc(sourcea.nchans * sizeof(float));

	//memset(diff_spectrum, 0x0, sourcea.nchans * sizeof(float));

    if(sourcea.nsamples != sourceb.nsamples) {
    	fprintf(stderr, "ERROR: sample count doesn't match! (sourcea: %ld, sourceb: %ld)\n", sourcea.nsamples, sourceb.nsamples);
    }

    long int candwidth;
    long int hitchan;
    long int left, right;
    float mean;
    candwidth = 512;
    
    for(i=0;i<sourcea.nchans;i++) diff_spectrum[i] = (sourcea.integrated_spectrum[i] - sourceb.integrated_spectrum[i])/sourceb.integrated_spectrum[i];


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
			  
	



	normalize(diff_spectrum, (long int) sourcea.nchans);

	candsearch_onoff(diff_spectrum, 512, 5, &sourcea, &sourceb);   
	
	
	


/* array will need to be grown */

/* time steps must be a power of 2 and edges will need padding */
/* padding on edges should be n_timesteps larger */



long int gulplength = sourcea.nsamples*sourcea.nchans;

sourcea.dimY = (long int) pow(2, ceil(log2(floor(sourcea.nsamples))));
fprintf(stderr, "new total time steps in dedoppler matrix: %ld\n", sourcea.dimY); 
sourcea.dimX = sourcea.nchans + (8 * sourcea.dimY);


if(padleft == NULL && padright == NULL) {
	padleft = (float*) malloc(4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	padright = (float*) malloc(4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	memset(padleft, 0x0, 4 * sourcea.dimY * sourcea.dimY * sizeof(float));
	memset(padright, 0x0, 4 * sourcea.dimY * sourcea.dimY * sizeof(float));
} 






if(ibrev == NULL) {

	/* build index mask for in-place tree doppler correction */	
	ibrev = (long int *) calloc(sourcea.dimY, sizeof(long int));
	drift_indexes = (int *) calloc(sourcea.nsamples, sizeof(int));
	
	for (i=0; i<sourcea.dimY; i++) {
		ibrev[i] = bitrev((long int) i, (long int) log2((double) sourcea.dimY));
	}

	/* solve for the indices of unique doppler drift rates */
	/* if we pad with zero time steps, we'll double up on a few drift rates as we step through */
	/* all 2^n steps */
	/* place a counter into dedoppler array in the last valid spectrum of the array */
	/* perform a dedoppler correction, then identify the indices corresponding to unique values of the counter */
	
	for(i=0;i<sourcea.dimY;i++) {
		tree_dedoppler[sourcea.dimX * (sourcea.nsamples - 1) + i] = i; 
	}
	
	taylor_flt(tree_dedoppler, sourcea.dimX, tsteps);

	
	k = -1;
	for(i=0;i<tsteps;i++){
	   //printf("De-doppler rate: %f Hz/sec\n", i);
	   indx  = (ibrev[i] * tdwidth);
		
	   for(j=0;j<1;j++){
			if(tree_dedoppler[indx+j] != k) {
				k = tree_dedoppler[indx+j];
				drift_indexes[k]=i;
				//printf("time index: %02d Sum: %02f, ", i, tree_dedisperse[indx+j]);				
			}
	   }
	
	}

}




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








