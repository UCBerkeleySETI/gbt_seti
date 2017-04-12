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
#include <mysql.h>
#include "setimysql.h"


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



void filterbanksearch_print_usage();

int main(int argc, char *argv[]) {

	struct filterbank_input sourcea;	
	struct filterbank_input sourceb;
	
	sourcea.bucketname = NULL;
	/* enable doppler search mode */
	int dopplersearchmode = 0;

	float *diff_spectrum;
	if(argc < 2) {
		exit(1);
	}

	int c;
	long int i,j,k;
	float zscore = 15;
	opterr = 0;
 
 	sourcea.zapwidth = 1;
	sourceb.zapwidth = 1;

	sourcea.candwidth = 512;
	sourceb.candwidth = 512;
	sourcea.filename = NULL;
	sourceb.filename = NULL;
    sourcea.Xpadframes = 0;
	sourceb.Xpadframes = 0;
 
	while ((c = getopt (argc, argv, "Vvdhf:b:s:a:s:z:d:")) != -1)
	  switch (c)
		{
		case 'h':
		  filterbanksearch_print_usage();
		  exit(0);
		  break;
		case 'a':
		  sourcea.filename = optarg;
		  break;
		case 'b':
		  sourceb.filename = optarg;
		  break;
		case 'd':
		  dopplersearchmode = 1;
		  break;
		case 'z':
		  zscore = (float) atof(optarg);
		  break;
		case 's':
		  sourcea.bucketname = optarg;
		  sourceb.bucketname = optarg;
		  break;
		case 'f':
		  sourcea.folder = optarg;
		  sourceb.folder = optarg;
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
	
	//dbconnect(sourcea.conn);
	
	sourcea.inputfile = fopen(sourcea.filename, "rb");
	read_filterbank_header(&sourcea);
	
	/* guess number of polyphase channels */
	sourcea.polychannels = (long int) round(fabs((double) sourcea.nchans * (double) sourcea.foff)/(187.5/64)); 
	sourceb.polychannels = sourcea.polychannels;
	
	fprintf(stderr, "%d %f Polyphase channels %ld\n", sourcea.nchans, sourcea.foff, sourcea.polychannels);

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
	candsearch_onoff(diff_spectrum, 512, zscore, &sourcea, &sourceb);   

    for(i=0;i<sourcea.nchans;i++) diff_spectrum[i] = (sourceb.integrated_spectrum[i] - sourcea.integrated_spectrum[i])/sourcea.integrated_spectrum[i];
	normalize(diff_spectrum, (long int) sourceb.nchans);
	candsearch_onoff(diff_spectrum, 512, zscore, &sourceb, &sourcea);   


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








