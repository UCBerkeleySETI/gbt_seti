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
#include <fftw3.h>
#include <sys/stat.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_multifit.h>

/* Guppi channel-frequency mapping */
/* sample at 1600 MSamp for an 800 MHz BW */
/* N = 256 channels with frequency centers at m * fs/N */
/* m = 0,1,2,3,... 255 */
//#include "filterbank.h"


/* prototypes */

float log2f(float x);
double log2(double x);
long int lrint(double x);

int exists(const char *fname);
double coeff(int *xarray, double *yarray, long long int num, long long int length, int degree, double *fittedvals);
void print_usage(char *argv[]); 
int channelize(unsigned char *channelbuffer, float *spectra, long long int numsamples, int fftlen);

void taylor_flt(float outbuf[], long int mlen, long int nchn);
void  FlipX(float  Outbuf[], long int xdim, long int ydim);
void FlipBand(float  Outbuf[], long int nchans, long int NTSampInRead);
void AxisSwap(float Inbuf[], float Outbuf[], long int nchans, long int NTSampInRead);
long int bitrev(long int inval,long int nbits);
double chan_freq(double center_freq, double channel_bw, double band_size, long long int fftlen, long int coarse_channel, long int fine_channel, long int tdwidth);

void imswap4 (char *string, int nbytes);
void simple_fits_write (FILE *fp, float *vec, int tsteps_valid, int freq_channels);


/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);




struct gpu_input {
	char *file_prefix;
	struct guppi_params gf;
	struct psrfits pf;	
	unsigned int filecnt;
	FILE *fil;
	int invalid;
	int curfile;
	int overlap;   /* add this keyword here since it doesn't seem to appear in guppi_params.c */
	long int first_file_skip; /* in case there's 8bit data in the header of file 0 */
};

float quantlookup[4];
int ftacc;
int spectraperint;
int overlap; // amount of overlap between sub integrations in samples

int N = 128;



fftwf_complex *in, *out;
fftwf_plan p;                 

int main(int argc, char *argv[]) {


	int filecnt=0;
    char buf[32768];
	

	unsigned char *channelbuffer;
	float *spectra;
 	float *spectrum;
 	float *spectrum_sum;
 	
	float *bandpass;
	
	
	char filname[250];

	struct gpu_input rawinput;	
	
	long long int startindx;
	long long int curindx;
	long long int chanbytes=0;
	long long int chanbytes_overlap = 0;
	long long int subint_offset = 0;
	long long int channelbuffer_pos = 0;

	long int nframes = 0;
	long int nchans = 0;
	
	int indxstep = 0;
	int channel = -1;


	/* de-doppler variables */
    long int tsteps, tsteps_valid;
    float *tree_dedoppler;
    float *tree_dedoppler_flip;
	char *zero_mask;
	int *drift_indexes;
	long int *ibrev;
	long int indx;
	double drift_rate_resolution;

	
	
	quantlookup[0] = 3.3358750;
   	quantlookup[1] = 1.0;
   	quantlookup[2] = -1.0;
   	quantlookup[3] = -3.3358750;

    
    
	size_t rv=0;
	long unsigned int by=0;
    
    FILE *partfil = NULL;  //output file
    FILE *bandpass_file = NULL;
    
	int c;
	long int i,j,k,m,n;
    int vflag=0; //verbose
    
    char *bandpass_file_name=NULL;
    char *partfilename=NULL;
    char *scratchpath=NULL;


    
	rawinput.file_prefix = NULL;
	rawinput.fil = NULL;
	rawinput.invalid = 0;
	rawinput.first_file_skip = 0;
	
	long long int fftlen;
	fftlen = 32768;

	/* if writing to disk == 1 */
	char disk = 0;
	
    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "Vvdi:o:c:f:b:s:")) != -1)
         switch (c)
           {
           case 'c':
             channel = atoi(optarg);
             break;
           case 'v':
             vflag = 1;
             break;
           case 'd':
             disk = 1;
             break;             
           case 'f':
             fftlen = atoll(optarg); //length of fft over 3.125 MHz channel
             break;             
           case 'b':
             bandpass_file_name = optarg;
             break;
           case 's':
             scratchpath = optarg;
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'i':
			 rawinput.file_prefix = optarg;
             break;
           case 'o':
			 partfilename = optarg;
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



	


/* no input specified */
if(rawinput.file_prefix == NULL) {
	printf("WARNING no input stem specified%ld\n", (i+1));
	exit(1);
}

if(strstr(rawinput.file_prefix, ".0000.raw") != NULL) memset(rawinput.file_prefix + strlen(rawinput.file_prefix) - 9, 0x0, 9);

char tempfilname[250];





/* set file counter to zero */
j = 0;
struct stat st;
long int size=0;
do {
	sprintf(filname, "%s.%04ld.raw",rawinput.file_prefix,j);
	printf("%s\n",filname);		
	j++;
	if(exists(filname)) { 
		stat(filname, &st);
		size = size + st.st_size;
	}
} while (exists(filname));
rawinput.filecnt = j-1;
printf("File count is %i  Total size is %ld bytes\n",rawinput.filecnt, size);



/* didn't find any files */
if(rawinput.filecnt < 1) {
	printf("no files for stem %s found\n",rawinput.file_prefix);
	exit(1);		
}

/* didn't select a channel */
if(channel < 0) {
	printf("must specify a channel!\n");
	exit(1);		
}


fprintf(stderr, "Will channelize spectra of length %lld pts\n", fftlen);
fprintf(stderr, "outputing average spectra to %s\n", partfilename);


/* open the first file for input */
sprintf(filname, "%s.0000.raw", rawinput.file_prefix);
rawinput.fil = fopen(filname, "rb");



/* if we managed to open a file */
if(rawinput.fil){
	  if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768){
		   
		   
		   guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);
		   
		   if(rawinput.pf.hdr.nbits == 8) {
			  
			  fprintf(stderr, "got an 8bit header\n");
			  
			  /* figure out the size of the first subint + header */
		      rawinput.first_file_skip = rawinput.pf.sub.bytes_per_subint + gethlength(buf);
				
			  /* rewind to the beginning */	
		   	  fseek(rawinput.fil, -32768, SEEK_CUR);
		   	  
		   	  /* seek past the first subint + header */
		   	  fseek(rawinput.fil, rawinput.first_file_skip, SEEK_CUR);

			  /* read the next header */
		   	  fread(buf, sizeof(char), 32768, rawinput.fil);
			  guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);
			  fclose(rawinput.fil);

		   } else {
 
			  fclose(rawinput.fil);
		   }

		   /* we'll use this file to set the params for the whole observation */
		   
		   rawinput.fil = NULL;

		   hgeti4(buf, "OVERLAP", &rawinput.overlap);

			
		   fprintf(stderr, "packetindex %lld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
		   fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
		   fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
		   fprintf(stderr, "bytes_per_subint: %d\n\n",rawinput.pf.sub.bytes_per_subint);

		   if (rawinput.pf.sub.data) free(rawinput.pf.sub.data);
		   
		   rawinput.pf.sub.data  = (unsigned char *) malloc(rawinput.pf.sub.bytes_per_subint);
			
	  } else {
	  		printf("couldn't read a header\n");
			exit(1);
	  }
} else {
	printf("couldn't open first file\n");
	exit(1);
}

channelbuffer  = (unsigned char *) calloc( lrint( (double) size/((double) rawinput.pf.hdr.nchan)), sizeof(char) );
printf("malloc'ing %ld Mbytes for processing channel %d\n",  lrint( (double) size/((double) rawinput.pf.hdr.nchan)), channel );	


//	tstart=band[first_good_band].pf.hdr.MJD_epoch;
//	tsamp = band[first_good_band].pf.hdr.dt * ftacc;

//	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
//	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));

	


printf("calculating index step\n");

/* number of packets that we *should* increment by */
indxstep = (int) ((rawinput.pf.sub.bytes_per_subint * 4) / rawinput.gf.packetsize) - (int) (rawinput.overlap * rawinput.pf.hdr.nchan * rawinput.pf.hdr.rcvr_polns * 2 / rawinput.gf.packetsize);


//spectraperint = indxstep * band[first_good_band].gf.packetsize / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 * ftacc);
//spectraperint = ((rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan) - rawinput.overlap) / ftacc;	

nchans = rawinput.pf.hdr.nchan;
overlap = rawinput.overlap;

/* number of non-overlapping bytes in each channel */
/* indxstep increments by the number of unique packets in each sub-integration */
/* packetsize is computed based on the original 8 bit resolution */
/* divide by 4 to get to 2 bits, nchan to get to number of channels */
chanbytes = indxstep * rawinput.gf.packetsize / (4 * rawinput.pf.hdr.nchan); 
printf("chan bytes %lld\n", chanbytes);


/* total number of bytes per channel, including overlap */
chanbytes_overlap = rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan;


/* memory offset for our chosen channel within a subint */
subint_offset = channel * chanbytes_overlap;



printf("Index step: %d\n", indxstep);
printf("bytes per subint %d\n",rawinput.pf.sub.bytes_per_subint );







fflush(stdout);


/* check to see if scratch is empty */
if(disk) {
  
	for(i = 0; i < nchans; i++) {
	   sprintf(tempfilname, "%s/guppi%ld.bin",scratchpath,i);
	   
	   if(exists(tempfilname)){
			fprintf(stderr, "Error!  %s exists...  proceeding to file processing... this is a debug mode\n", tempfilname);
			stat(tempfilname, &st);
			channelbuffer_pos = st.st_size;
	   		rawinput.invalid = 1;
	   		//exit(1);
	   }
	
	}

}


startindx = rawinput.gf.packetindex;
curindx = startindx;

filecnt = rawinput.filecnt;

rawinput.curfile = 0;			

do{
										
	if(!rawinput.invalid){						  
		  if(rawinput.fil == NULL) {
			  /* no file is open for this band, try to open one */
			  sprintf(filname, "%s.%04d.raw",rawinput.file_prefix,rawinput.curfile);
			  printf("filename is %s\n",filname);
			  if(exists(filname)){
				 printf("opening %s\n",filname);				
				 rawinput.fil = fopen(filname, "rb");			 
				 if(rawinput.curfile == 0 && rawinput.first_file_skip != 0) fseek(rawinput.fil, rawinput.first_file_skip, SEEK_CUR);  
			  }	else {
			  	rawinput.invalid = 1;
		  	  	printf("couldn't open any more files!\n");
		  	  	//exit(1);
		  	  }
		  }

		  if(rawinput.fil){
				if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768) {
				
					fseek(rawinput.fil, -32768, SEEK_CUR);
					if(vflag>=1) fprintf(stderr, "header length: %d\n", gethlength(buf));
					guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);
					if(vflag>=1) {
						 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
						 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
						 fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
						 fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
						 fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
					}
					
			   		if(rawinput.gf.packetindex == curindx) {
						 /* read a subint with correct index */
						 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
		
						 if((long int)rv == rawinput.pf.sub.bytes_per_subint){
							if(vflag>=1) fprintf(stderr,"read %d bytes from %ld in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
						 	

							if(vflag>=1) fprintf(stderr, "buffer position: %lld\n", channelbuffer_pos);
						 							 	
						 	if(disk) {
						 		if(vflag>=1) fprintf(stderr, "dumping to disk...\n");
						 		channels_to_disk(rawinput.pf.sub.data, scratchpath, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);
							} else {
								memcpy(channelbuffer + channelbuffer_pos, rawinput.pf.sub.data + subint_offset, chanbytes);				
							}

						 	
						 	channelbuffer_pos = channelbuffer_pos + chanbytes;

						 } else {
						 	 rawinput.fil = NULL;
							 fprintf(stderr,"ERR: couldn't read as much as the header said we could... assuming corruption and exiting...\n");
							 exit(1);
						 }
						 
					
					} else if(rawinput.gf.packetindex > curindx) {
						 fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						 /* read a subint with too high an indx, must have dropped a whole subintegration*/
				

						/* pf.sub.data *should* still contain the last valid subint */
						if(disk) {
						 	if(vflag>=1) fprintf(stderr, "dumping to disk...\n");
						 	channels_to_disk(rawinput.pf.sub.data, scratchpath, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);
						} else {
							/* grab a copy of the last subint  - probably should add gaussian noise here, but this is better than nothing! */
							memmove(channelbuffer + channelbuffer_pos, channelbuffer + channelbuffer_pos - chanbytes, chanbytes);						
						}
	
						/* increment buffer position to cover the dropped subint */
						channelbuffer_pos = channelbuffer_pos + chanbytes;

						/* We'll get the current valid subintegration again on the next time through this loop */

						

					} if(rawinput.gf.packetindex < curindx) {
						 fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						 /* somehow we were expecting a higher packet index than we got !?!? */
	
						 /* try to read an extra spectra ahead */
						 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 

						 if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768) {
							  
							  fseek(rawinput.fil, -32768, SEEK_CUR);
							  guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);	 

							  if(rawinput.gf.packetindex == curindx) {
							  	  fprintf(stderr,"synced back up...\n");

								  fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
								  rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
				 
								  if((long int)rv == rawinput.pf.sub.bytes_per_subint){
									 fprintf(stderr,"Read %d more bytes from %d in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);

		 
									 if(vflag>=1) fprintf(stderr, "buffer position: %lld\n", channelbuffer_pos);
									 
									 if(disk) {
						 				if(vflag>=1) fprintf(stderr, "dumping to disk...\n");
						 				channels_to_disk(rawinput.pf.sub.data, scratchpath, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);
									 } else {
										 memcpy(channelbuffer + channelbuffer_pos, rawinput.pf.sub.data + subint_offset, chanbytes);		 									 
									 }
								  
									 channelbuffer_pos = channelbuffer_pos + chanbytes;				  
								  } else {
									 fprintf(stderr,"couldn't read as much as the header said we could... assuming corruption and exiting...\n");
									 exit(1);
								  }

								} else {
								  fprintf(stderr,"ERROR! skipped an extra packet, still off\n");
								  /* this shouldn't happen very often....  but if it does, something is seriously wrong. */
								  exit(1);
								}
						 } else {
							  /* file open but couldn't read 32KB */
							  fclose(rawinput.fil);
							  rawinput.fil = NULL;
							  rawinput.curfile++;														 								 
						 }								 
					
					}

				} else {

				/* file open but couldn't read 32KB */
				   fclose(rawinput.fil);
				   rawinput.fil = NULL;
				   rawinput.curfile++;						
				}
		  }			 	 	 
	}

										
	if(rawinput.fil != NULL) curindx = curindx + indxstep;


} while(!(rawinput.invalid));
	
	


	
	fprintf(stderr, "finishing up...\n");

	if(vflag>=1) fprintf(stderr, "bytes: %ld\n",by);
	

	fprintf(stderr, "copied %lld bytes\n", channelbuffer_pos);
	
	if (rawinput.pf.sub.data) {
		 free(rawinput.pf.sub.data);
		 fprintf(stderr, "freed subint data buffer\n");
	}

    nframes = (long int) floor(channelbuffer_pos/fftlen);


int nfiles;
if(disk) { nfiles = nchans; } else { nfiles = 1; }

	double stddev;
	double mean;

	double max_search_rate = 50; //Maximum doppler drift rate to search in Hz/sec
	double obs_length;
	float candthresh = 25.0;
	
	float *padleft = NULL;
	float *padright = NULL;


	long long int tdwidth;

	float *candidatedata;
	FILE *candidatefile;
	char candidatefilename[255];
	
	
tsteps = (long int) pow(2, ceil(log2(floor(channelbuffer_pos/fftlen))));
fprintf(stderr, "total time steps in dedoppler matrix: %ld\n", tsteps); 

tsteps_valid = (long int) floor(channelbuffer_pos/fftlen);
fprintf(stderr, "total valid time steps in dedoppler matrix: %ld\n", tsteps_valid); 
	
obs_length = tsteps_valid * fftlen * rawinput.pf.hdr.dt; // obs length in seconds
printf("obs length: %g\n", obs_length);

drift_rate_resolution = (1000000.0 * rawinput.pf.hdr.df) / ( ((double) fftlen) * obs_length); // Hz/sec - guppi chan bandwidth is in MHz
printf("DR resolution: %g\n", drift_rate_resolution);

printf("nominal maximum drift rate to be searched %g\n", drift_rate_resolution * tsteps_valid);




for (m = 0; m < nfiles; m++) {

	spectra = (float *) malloc (channelbuffer_pos * sizeof(float));


	
	/* perform channelization operation */

	if(disk) {
		if(readbin(m, channelbuffer, channelbuffer_pos, scratchpath) != 1){
			   fprintf(stderr, "Error! couldn't read scratch file\n");
			   exit(1);
			} 	
	
		channel = m;

/*
		if(m==7) {
		 for(i=channelbuffer_pos - 100;i<channelbuffer_pos;i++){
			 printf("%d 0x%02x\n",i, channelbuffer[i]);
			 usleep(1000000);
		 }			
		 }
*/		 
	}

    nframes = (long int) floor(channelbuffer_pos/fftlen);


	channelize(channelbuffer, spectra, channelbuffer_pos, fftlen);
	
	if(bandpass_file_name != NULL) {
		 fprintf(stderr, "equalizing using bandpass model: %s\n", bandpass_file_name);
		 bandpass = (float *) malloc (fftlen * sizeof(float));
		 bandpass_file = fopen(bandpass_file_name, "rb");
		 fread(bandpass, sizeof(float),fftlen, bandpass_file);
		 /* equalize all spectra */
		 for(i = 0; i < nframes; i++){	    
			 for(j = 0; j < fftlen; j++) spectra[i * fftlen + j] = spectra[i * fftlen + j] / bandpass[j] * 10000000;
		 }
		 free(bandpass);
		 fclose(bandpass_file);
	}

  	long int decimate_factor=0;
	if(drift_rate_resolution * tsteps_valid > max_search_rate) {
		printf("nominal max drift rate greater than allowed.... decimating.\n");
		decimate_factor = floor((double) tsteps_valid / (max_search_rate / drift_rate_resolution));	
	    for(i=0; (i * decimate_factor) < (tsteps_valid - decimate_factor); i++){
			fprintf(stderr, "i: %d ", i);
			for(k=1;k<decimate_factor;k++){
				for(j=0;j<fftlen;j++) spectra[(i*fftlen) + j] = spectra[(i*fftlen) + j] + spectra[(i*decimate_factor*fftlen) + (k*fftlen) + j];
	    	}
	    
	    
	    }
	
	tsteps_valid = i;
	nframes = tsteps_valid;
	tsteps = (long int) pow(2, ceil(log2(floor(tsteps_valid))));
	fprintf(stderr, "new total time steps in dedoppler matrix: %ld\n", tsteps); 

	fprintf(stderr, "new total valid time steps in dedoppler matrix: %ld\n", tsteps_valid); 

	}





	if(padleft == NULL && padright == NULL) {
		padleft = (float*) malloc(tsteps * 2 * tsteps_valid * sizeof(float));
		padright = (float*) malloc(tsteps * 2 * tsteps_valid * sizeof(float));
		memset(padleft, 0x0, tsteps * 2 * tsteps_valid * sizeof(float));
		memset(padright, 0x0, tsteps * 2 * tsteps_valid * sizeof(float));
	} 


	//if candidatedata matrix hasn't been allocated, allocate it.  We'll store a temporary vector 
	//of samples around each candidate signal here before creating a fits file.
	
	if(candidatedata == NULL) candidatedata = (float *) malloc(tsteps_valid * tsteps * 4 * sizeof(float));

	
	drift_indexes = (int *) calloc(tsteps_valid, sizeof(int));



	tdwidth = fftlen + (4 * tsteps);

	spectrum = (float *) malloc (tdwidth * sizeof(float));

	spectrum_sum = (float *) malloc (fftlen * sizeof(float));


	memset(spectrum, 0x0, tdwidth * sizeof(float));
	memset(spectrum_sum, 0x0, fftlen * sizeof(float));



	/* allocate array for dedopplering */
	tree_dedoppler = (float*) malloc(tsteps * tdwidth * sizeof(float));

	/* init dedopplering array to zero */
	memset(tree_dedoppler, 0x0, sizeof(float) * tsteps * tdwidth);  //initialize dedoppler correction arrays




	/* build index mask for in-place tree doppler correction */
	ibrev = (long int *) calloc(tsteps, sizeof(long int));
	
	for (i=0; i<tsteps; i++) {
		ibrev[i] = bitrev((long int) i, (long int) log2((double) tsteps));
	}

	/* solve for the indices of unique doppler drift rates */
	/* if we pad with zero time steps, we'll double up on a few drift rates as we step through */
	/* all 2^n steps */
	for(i=0;i<tsteps;i++) {
		tree_dedoppler[tdwidth * (tsteps_valid - 1) + i] = i; 
	}
	
	taylor_flt(tree_dedoppler, tsteps * tdwidth, tsteps);

	
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


	 /* sum all spectra */
	 for(i = 0; i < nframes; i++){	    
		 for(j = 0; j < fftlen; j++) spectrum_sum[j] = spectrum_sum[j] + spectra[i * fftlen + j];
	 }



	for(i=0;i<nframes;i++){
		for(j=0;j<(tsteps*2);j++) tree_dedoppler[tdwidth*i+j] = padleft[(i * (tsteps*2)) + j];
		for(j=fftlen + (tsteps * 2);j<(tdwidth);j++) tree_dedoppler[tdwidth*i+j] = padright[(i * (tsteps*2)) + j];
		for(j=(tsteps*2);j<(tdwidth - (tsteps*2));j++) tree_dedoppler[i*tdwidth+j] = spectra[(fftlen*i) + (j-(tsteps*2))]; 
	}


	//load end of current spectra into left hand side of next spectra 
	for(i=0;i<nframes;i++){
		for(j=0;j<(tsteps * 2);j++) padleft[(i * (tsteps*2)) + j] = spectra[(fftlen*(i+1)) - (tsteps * 2) + j];
	}

	/* copy spectra into new array */
  	//memcpy(tree_dedoppler, spectra, nframes * fftlen * sizeof(float));
	

	/* free original array */
	free(spectra);	


	/* allocate array for negative doppler rates */
	tree_dedoppler_flip = (float*) malloc(tsteps * tdwidth * sizeof(float));

	/* populate neg doppler array */
	memcpy(tree_dedoppler_flip, tree_dedoppler, tsteps * tdwidth * sizeof(float));


/* fold over the last 2 * tsteps samples with the preceding 2 * tsteps samples. */
//memcpy(spectrum + tdwidth - (2*tsteps), tree_dedoppler + indx + fftlen - (4 * tsteps), 2*tsteps * sizeof(float));



	fprintf(stderr, "Doppler correcting forward...\n");
	taylor_flt(tree_dedoppler, tsteps * tdwidth, tsteps);
	fprintf(stderr, "done...\n");


	for(k=0;k<tsteps_valid;k++) {
		indx  = (ibrev[drift_indexes[k]] * tdwidth);

		/* SEARCH POSITIVE DRIFT RATES */
		memcpy(spectrum, tree_dedoppler + indx, tdwidth * sizeof(float));
		comp_stats(&mean, &stddev, spectrum, tdwidth, 2*tsteps);


		/* normalize */
		for(i=0;i<tdwidth;i++) spectrum[i] = spectrum[i] - mean;
		for(i=0;i<tdwidth;i++) spectrum[i] = spectrum[i] / stddev;		
		
		//m = 0, 2t -> -4t forward
		//m = 0, 2t -> -4t reverse
		//m = 1,2,3 ... 31  0 -> -4t forward
		//m = 1,2,3 ... 31  2t -> -2t reverse
				
		if(m == 0) {

			  /* candidate search! */
			  for(i=(tsteps*2);i<tdwidth - (tsteps * 4);i++) {				   
				  if(spectrum[i] > candthresh) {
					   printf("candidate! frequency: %15.15g MHz drift: %g Hz/sec  SNR: %g  chan: %ld  drift: %ld\n", chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, i, tdwidth), drift_rate_resolution * k,spectrum[i], i, k);

					   memset(candidatedata, 0x0, tsteps_valid * tsteps * 4 * sizeof(float));

					   if(i >= (tsteps*2)) {
					   	   
					   	   for(n=0;n<tsteps_valid;n++) {
						   	   for(j=(i-(tsteps*2));j<(i+(tsteps*2));j++) candidatedata[ (n * tsteps * 4) + (j - (i-(tsteps*2)))] = tree_dedoppler_flip[(n * tdwidth) + j];
						   }
					   
					   } else {
					   	   for(n=0;n<nframes;n++) {
						   	   for(j=0;j<(i+(tsteps*2));j++) candidatedata[(n * tsteps * 4) + (j + ((tsteps*2) - i))] = tree_dedoppler_flip[(n * tdwidth) + j];
						   }					   
					   }
					   
					   
					   printf("done\n");

					   sprintf(candidatefilename, "%g_%15.15g_+%g.fits", spectrum[i], chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, i, tdwidth), drift_rate_resolution * k);

					   printf("%s\n", candidatefilename);
					   candidatefile = fopen(candidatefilename, "w");
					   printf("file open\n");

					   simple_fits_write(candidatefile, candidatedata, (int) tsteps_valid, (int) (4 * tsteps));
					   printf("fits written\n");

					   fclose(candidatefile);
					   
				  }
			  }


		} else {

			  /* candidate search! */
			  for(i=0;i<tdwidth - (tsteps * 4);i++) {				   
				  if(spectrum[i] > candthresh) {

					   printf("candidate! frequency: %15.15g MHz drift: %g Hz/sec  SNR: %g  chan: %ld  drift: %ld\n", chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, i, tdwidth), drift_rate_resolution * k,spectrum[i], i, k);

					   memset(candidatedata, 0x0, tsteps_valid * tsteps * 4 * sizeof(float));

					   if(i >= (tsteps*2)) {
					   	   
					   	   for(n=0;n<tsteps_valid;n++) {
						   	   for(j=(i-(tsteps*2));j<(i+(tsteps*2));j++) candidatedata[ (n * tsteps * 4) + (j - (i-(tsteps*2)))] = tree_dedoppler_flip[(n * tdwidth) + j];
						   }
					   
					   } else {
					   	   for(n=0;n<nframes;n++) {
						   	   for(j=0;j<(i+(tsteps*2));j++) candidatedata[(n * tsteps * 4) + (j + ((tsteps*2) - i))] = tree_dedoppler_flip[(n * tdwidth) + j];
						   }					   
					   }
					   
					   
					   printf("done\n");

					   sprintf(candidatefilename, "%g_%15.15g_+%g.fits", spectrum[i], chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, i, tdwidth), drift_rate_resolution * k);

					   printf("%s\n", candidatefilename);
					   candidatefile = fopen(candidatefilename, "w");
					   printf("file open\n");

					   simple_fits_write(candidatefile, candidatedata, (int) tsteps_valid, (int) (4 * tsteps));
					   printf("fits written\n");

					   fclose(candidatefile);


				  }
			  }
		
		}


	}


	/* copy back original array */
	memcpy(tree_dedoppler, tree_dedoppler_flip, tsteps * tdwidth * sizeof(float));

	/* Flip matrix across X dimension to search negative doppler drift rates */
	FlipX(tree_dedoppler_flip, tdwidth, tsteps);

	fprintf(stderr, "Doppler correcting reverse...\n");	
	taylor_flt(tree_dedoppler_flip, tsteps * tdwidth, tsteps);
	fprintf(stderr, "done...\n");



	for(k=0;k<tsteps_valid;k++) {
		indx  = (ibrev[drift_indexes[k]] * tdwidth);
		

		/* SEARCH NEGATIVE DRIFT RATES */
		memcpy(spectrum, tree_dedoppler_flip + indx, tdwidth * sizeof(float));
		comp_stats(&mean, &stddev, spectrum, tdwidth, 2*tsteps);


		/* normalize */
		for(i=0;i<tdwidth;i++) spectrum[i] = spectrum[i] - mean;
		for(i=0;i<tdwidth;i++) spectrum[i] = spectrum[i] / stddev;		
		
		//m = 0, 2t -> -4t forward
		//m = 0, 2t -> -4t reverse
		//m = 1,2,3 ... 31  0 -> -4t forward
		//m = 1,2,3 ... 31  2t -> -2t reverse
				
		if(m == 0) {

			  /* candidate search! */
			  for(i=(tsteps*2);i<tdwidth - (tsteps * 4);i++) {				   
				  if(spectrum[i] > candthresh) {
					   printf("candidate! frequency: %15.15g MHz drift: - %g Hz/sec  SNR: %g  chan: %ld  drift: %ld\n", chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, (tdwidth - 1 - i), tdwidth), drift_rate_resolution * k,spectrum[i], i, k);
					   //printf("%15.15f %15.15f\n", (rawinput.pf.hdr.fctr - 50) + ((i * rawinput.pf.hdr.df)/fftlen) + (channel * rawinput.pf.hdr.df), spectrum[i]);
				  }
			  }


		} else {

			  /* candidate search! */
			  for(i=(tsteps*2);i<tdwidth - (tsteps * 2);i++) {				   
				  if(spectrum[i] > candthresh) {
					   printf("candidate! frequency: %15.15g MHz drift: - %g Hz/sec  SNR: %g  chan: %ld  drift: %ld\n", chan_freq(rawinput.pf.hdr.fctr, rawinput.pf.hdr.df, rawinput.pf.hdr.BW, fftlen, channel, (tdwidth - 1 - i), tdwidth), drift_rate_resolution * k,spectrum[i], i, k);
					   //printf("%15.15f %15.15f\n", (rawinput.pf.hdr.fctr - 50) + ((i * rawinput.pf.hdr.df)/fftlen) + (channel * rawinput.pf.hdr.df), spectrum[i]);
				  }
			  }
		
		}

		
	}

	indx  = (ibrev[drift_indexes[0]] * tdwidth); //get index for dedoppler = 0 Hz/sec drift rate


	if(m==0) {	
		 partfil = fopen(partfilename, "w");	
		 for(j=(tsteps*2);j<(tdwidth-(tsteps*2));j++) fprintf(partfil, "%15.15f %15.15f %15.15f %15.15f\n", (rawinput.pf.hdr.fctr - 50) + ((j * rawinput.pf.hdr.df)/fftlen) + (channel * rawinput.pf.hdr.df), tree_dedoppler[indx+j], spectrum[j], spectrum_sum[j- (tsteps*2)]);
		 for(j=(tsteps*2);j<(tdwidth-(tsteps*2));j++) fprintf(partfil, "%15.15f %15.15f %15.15f %15.15f\n", (rawinput.pf.hdr.fctr - 50) + ((j * rawinput.pf.hdr.df)/fftlen) + (channel * rawinput.pf.hdr.df), tree_dedoppler_flip[indx+j], spectrum[j], spectrum_sum[j- (tsteps*2)]);

		 fclose(partfil);
	}
	
	free(tree_dedoppler_flip);
	free(tree_dedoppler);
	free(ibrev);
	free(drift_indexes);
	free(spectrum);
	free(spectrum_sum);
}
	/* subtract mean */
	/* divide by rms */

	free(channelbuffer);
	


	
	zero_mask = (char*) calloc(fftlen, sizeof(char));
	
	
		
	indx  = (ibrev[0] * fftlen); //get index for dedoppler = 0 Hz/sec drift rate

	fprintf(stderr, "0 Hz/sec index: %ld\n", indx);



	exit(1);
	
	
	
	
	partfil = fopen(partfilename, "w");	
	for(j = 0; j < fftlen; j++) {		
		fprintf(partfil, "%15.15f %15.15f\n", (rawinput.pf.hdr.fctr - 50) + ((j * rawinput.pf.hdr.df)/fftlen) + (channel * rawinput.pf.hdr.df)  , spectrum[j]);		
	}
	fclose(partfil);


	fprintf(stderr, "closed output file...\n");


	fftwf_destroy_plan(p);
	fftwf_free(out);
	
	fprintf(stderr, "cleaned up FFTs...\n");

    exit(1);


}






void bin_print_verbose(unsigned char x)
/* function to print decimal numbers in verbose binary format */
/* x is integer to print, n_bits is number of bits to print */
{

   int j;
   printf("no. 0x%08x in binary \n",(int) x);

   for(j=8-1; j>=0;j--){
	   printf("bit: %i = %i\n",j, (x>>j) & 01);
   }

}



/* we'll need to pass the channel array,  */

int channelize(unsigned char *channelbuffer, float *spectra, long long int numsamples, int fftlen)
{
int i,j,k;

/* number of fft frames in channel buffer */
long long int numframes;

numframes = (long long int) floor(numsamples/fftlen);

fftwf_complex *in;
fftwf_complex *out;
fftwf_plan plan_forward;

if(!fftwf_import_wisdom_from_filename("/home/siemion/sw/kepler/treedop_wisdom.txt")) printf("Couldn't read wisdom!\n");

in = fftwf_malloc ( sizeof ( fftwf_complex ) * fftlen );
out = fftwf_malloc ( sizeof ( fftwf_complex ) * fftlen );


fprintf(stderr, "planning fft...\n");

plan_forward = fftwf_plan_dft_1d ( fftlen, in, out, FFTW_FORWARD, FFTW_PATIENT);

fprintf(stderr, "executing fft over %lld frames \n", numframes);

/* unpack one fft frame worth of samples */

/* do the transform */

/* detect and sum */

for(i = 0; i < numframes; i++) {

	for(k = 0; k < fftlen; k++) {
		in[k][0] = (channelbuffer[ k + (i * fftlen) ] >> (0 * 2) & 1) +  (2 * (channelbuffer[ k + (i * fftlen) ] >> (0 * 2 + 1) & 1)); //real pol 0
		in[k][1] = (channelbuffer[ k + (i * fftlen) ] >> (1 * 2) & 1) +  (2 * (channelbuffer[ k + (i * fftlen) ] >> (1 * 2 + 1) & 1)); //imag pol 0
	}

   	fftwf_execute ( plan_forward );

	for(k=0;k<fftlen;k++){
		spectra[ (k) + (i * fftlen) ] = powf(out[(k+fftlen/2)%fftlen][0],2) + powf(out[(k+fftlen/2)%fftlen][1],2);  /*real^2 + imag^2 for pol 0 */
	}	

	for(k = 0; k < fftlen; k++) {
		in[k][0] = (channelbuffer[ k + (i * fftlen) ] >> (2 * 2) & 1) +  (2 * (channelbuffer[ k + (i * fftlen) ] >> (2 * 2 + 1) & 1)); //real pol 1
		in[k][1] = (channelbuffer[ k + (i * fftlen) ] >> (3 * 2) & 1) +  (2 * (channelbuffer[ k + (i * fftlen) ] >> (3 * 2 + 1) & 1)); //imag pol 1
	}

    fftwf_execute ( plan_forward );

	for(k=0;k<fftlen;k++){
		spectra[ (k) + (i * fftlen) ] = spectra[ (k) + (i * fftlen) ] + powf(out[(k+fftlen/2)%fftlen][0],2) + powf(out[(k+fftlen/2)%fftlen][1],2);  /*real^2 + imag^2 for pol 1 */
	}
	

	/* set the DC channel equal to the mean of the two adjacent channels */
	//fprintf(stderr, "dividing\n");
	spectra[ (fftlen/2) + (i * fftlen) ] = (spectra[ (fftlen/2) + (i * fftlen) - 1 ] + spectra[ (fftlen/2) + (i * fftlen) + 1])/2;
	
			

}	


fftwf_free(in);
fftwf_free(out);

return 0;
}



int exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return 1;
    }
    return 0;
}

int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) {
    return 1;
  } else {
    return 0;
  }
}

void error_message(char *message) /*includefile */
{
  fprintf(stderr,"ERROR: %s\n",message);
  exit(1);
}

double coeff(int *xarray, double *yarray, long long int num, long long int length, int degree, double *fittedvals)
{
  int i,j; 
  int n;
  double xi, yi, ei, chisq, variance=0, average=0, sigma=0;
  gsl_matrix *X, *cov;
  gsl_vector *y, *w, *c;

  n = num;
  X = gsl_matrix_alloc (n, degree);
  y = gsl_vector_alloc (n);
  w = gsl_vector_alloc (n);

  c = gsl_vector_alloc (degree);
  cov = gsl_matrix_alloc (degree, degree);

  /* num is total valid points in xarray and yarray */
  /* length is the total length of the fittedvals vector */
  

  for (i = 0; i < num; i++)
    {

      xi = (double) xarray[i];
      yi = yarray[i];
	  
	  /* assign uniform weights to all values */
      ei = 1;
      //.001 * yarray[i];

      //if(i%1024 == 0) printf ("%g %g +/- %g\n", xi, yi, ei);

      gsl_matrix_set (X, i, 0, 1.0);      


      for(j=1;j<degree;j++){
		  gsl_matrix_set (X, i, j, pow(xi, j));

      }

      gsl_vector_set (y, i, yi);
      gsl_vector_set (w, i, 1.0/(ei*ei));

    }

  
    gsl_multifit_linear_workspace * work 
      = gsl_multifit_linear_alloc (n, degree);
    gsl_multifit_wlinear (X, w, y, c, cov,
                          &chisq, work);
    gsl_multifit_linear_free (work);
  

#define C(i) (gsl_vector_get(c,(i)))
#define COV(i,j) (gsl_matrix_get(cov,(i),(j)))

  
//    printf ("# best fit: Y = %g",C(0)); 
//    for(i=1;i<degree;i++) printf (" + %g X^%d", C(i), i); 
//	printf("\n");

for (j=0;j<length;j++)
{
	fittedvals[j] = C(0);
	for(i = 1; i < degree; i++){
		fittedvals[j] = fittedvals[j] + (C(i) * pow(j, i));
	}

}

for (j=0;j<num;j++)
{
   average = average + fittedvals[xarray[j]];
}

average = average / num;

for(j=0;j<num;j++) {
    variance = variance + pow( (fittedvals[xarray[j]] - yarray[j]), 2);
}
variance = variance / (n - 1);

sigma = pow(variance, 0.5);
//printf("sigma in func: %15.15f average: %15.15f\n", sigma, average);

	gsl_matrix_free(X);
	gsl_vector_free(y);
	gsl_vector_free(w);
	gsl_vector_free(c);
	gsl_matrix_free(cov);

return sigma;
}


void polyfit_norm(int fftlen, int nbytes, float *spectra) {

	double *polyfit;
	double mean = 0; 
	double rms = 0;  
	long long int pointcount = 0;
	int *xarray;
	double *yarray;
	double poly_stddev;
	int i,j,k;
	
	polyfit = malloc(sizeof(double) * fftlen);
	xarray = malloc(sizeof(int) * fftlen);
	yarray = malloc(sizeof(double) * fftlen);


	for(i = 0; i < (long long int) floor(nbytes/fftlen); i++){	    

		  mean = 0; rms = 0;
	  
		  /* calculate mean */
		  for(k=0;k<fftlen;k++) mean = mean + spectra[ (k) + (i * fftlen) ];
		  mean = mean / (double) fftlen;
	  
		  /* calculate rms */
		  for(k=0;k<fftlen;k++) rms = rms + powf((spectra[ (k) + (i * fftlen) ] - mean), 2);
		  rms = powf((rms/((float) fftlen - 1)), 0.5);
	  	  		  
	 	  pointcount = 0;
		  
		  for(k=0;k<fftlen;k++) {
			
			   if(spectra[i * fftlen + k] < (5 * mean) ) {				
				   yarray[pointcount] = spectra[ (k) + (i * fftlen) ];
			   	   xarray[pointcount] = k;
				   pointcount++;
			   }		  
		  
		  }
			
		  //fprintf(stderr, "ptcnt: %lld mean: %15.15f rms %15.15f\n", pointcount, mean, rms);

		  /* calulcate polynomial fit for residual baseline subtraction */
		  poly_stddev = coeff(xarray, yarray, pointcount, fftlen, 4, polyfit);

		  /* divide out by polynomial model */		 
		  for(k=0;k<fftlen;k++) spectra[ (k) + (i * fftlen) ] = spectra[ (k) + (i * fftlen) ] / polyfit[k];
		 			  
			/*
			partfil = fopen(partfilename, "w");	
			for(j = 0; j < fftlen; j++) {		
				fprintf(partfil, "%15.15f %15.15f\n", spectra[i * fftlen + j], polyfit[j]);		
			}
			fclose(partfil);
			exit(1);
			*/
	}	


}

void gauss_smooth() {

/*
	fprintf(stderr, "constructing Gaussian convolution kernel...\n");
	Gwidth = fftlen / 256;
	filter = (float *) malloc (fftlen * sizeof(float) + Gwidth);
	memset(filter, 0x0, fftlen * sizeof(float)+ Gwidth);

	stddev=Gwidth/7; // standard deviation

	for(i = 0; i<Gwidth/2; i++) {
		gaussval = -1 * (Gwidth/2 - i* (Gwidth/(Gwidth-1)));
		filter[i] = expf(-.5*((gaussval/stddev).^2));
	}
	
	
	for(i = Gwidth/2 - 1; i>=0; i--) {
		gaussval = Gwidth/2 - i * (Gwidth/(Gwidth-1));
 		filter[Gwidth - 1 - i] = expf(-.5*((gaussval/stddev).^2));
	}

	//expf(-.5*((gaussval/stddev).^2));  // create a normal distribution

	for(i = Gwidth/2 - 1; i>=0; i--) {
		gaussval = Gwidth/2 - i * (Gwidth/(Gwidth-1));
 		filter[Gwidth - 1 - i] = expf(-.5*((gaussval/stddev).^2));
	}
	
	// normalize window 
	for(i = 0; i<Gwidth; i++) gsum = gsum + filter[i];
	for(i = 0; i<Gwidth; i++) filter[i] = filter[i]./gsum;




	// sum all spectra, setting those > 8 * mean to the mean of the two adjacent channels 
	for(i = 0; i < (long long int) floor(channelbuffer_pos/fftlen); i++){
		for(j = 0; j < fftlen; j++) mean = mean + spectra[i * fftlen + j];
		mean = mean / fftlen;
		for(j = 0; j < fftlen; j++) {
			if(spectra[i * fftlen + j] > 8 * mean  && j > 0 && j < (fftlen - 1)) {				
				spectrum[j] = spectrum[j] + ((spectra[i * fftlen + j - 1] + spectra[i * fftlen + j + 1])/2);
			} else { 
				spectrum[j] = spectrum[j] + spectra[i * fftlen + j];
			}
		}	
	}
	
	
//	     fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out,
//                                    unsigned flags);
	
	 fftwf_complex *outspec;
	 fftwf_complex *outfilt;

	 fftwf_complex *smoothspec;

	 float *inputvector;
	
	 inputvector = (float *) malloc (fftlen * sizeof(float) + Gwidth);
	 memset(inputvector, 0x0, fftlen * sizeof(float)+ Gwidth);

	 
	 // create padded vector to avoid edge effects on convolution 
	 for(i = Gwidth; i<(Gwidth+fftlen); i++) inputvector[i] = (spectrum[i - Gwidth] / fftlen);
	 for(i = 0; i<Gwidth; i++) inputvector[i] = inputvector[Gwidth];


	 fftwf_plan ps;
	 fftwf_plan pf;
	 fftwf_plan rs;



	 ps = fftwf_plan_dft_r2c_1d(fftlen, spectrum, outdata, FFTW_PATIENT);
	 pf = fftwf_plan_dft_r2c_1d(fftlen, filter, outfilt, FFTW_PATIENT);
	 rs = fftwf_plan_dft_c2c_1d(fftlen, filter, outfilt, FFTW_BACKWARD, FFTW_PATIENT);


	 fftwf_execute ( ps );
	 fftwf_execute ( pf );
	 fftwf_execute ( rs );

	*/
	
}


void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input_prefix -c channel -p N\n", argv[0]);
	fprintf(stderr, "		N = 2^N FFT Points\n");
	fprintf(stderr, "		-v or -V for verbose\n");
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




void AxisSwap(float Inbuf[],
              float Outbuf[], 
              long int   nchans, 
              long int   NTSampInRead) {
  long int    j1, j2, indx, jndx;

  for (j1=0; j1<NTSampInRead; j1++) {
    indx  = (j1 * nchans);
    for (j2=(nchans-1); j2>=0; j2--) {
      jndx = (j2 * NTSampInRead + j1);
      Outbuf[jndx]  = Inbuf[indx+j2];
    }
  }

  return;
}




void  FlipBand(float  Outbuf[], 
               long int    nchans, 
               long int    NTSampInRead) {

  long int    indx, jndx, kndx, i, j;
  float *temp;

  temp  = (float *) calloc((NTSampInRead*nchans), sizeof(float));

  indx  = (nchans - 1);
  for (i=0; i<nchans; i++) {
    jndx = (indx - i) * NTSampInRead;
    kndx = (i * NTSampInRead);
    memcpy(&temp[jndx], &Outbuf[kndx], sizeof(float)*NTSampInRead);
  }
  memcpy(Outbuf, temp, (sizeof(float)*NTSampInRead * nchans));

  free(temp);

  return;
}


void  FlipX(float  Outbuf[], 
               long int    xdim, 
               long int    ydim) {

  long int    indx, jndx, kndx, i, j, revi;
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



/*  ======================================================================  */
/*  This is a function to Taylor-tree-sum a data stream. It assumes that    */
/*  the arrangement of data stream is, all points in first spectra, all     */
/*  points in second spectra, etc...  Data are summed across time           */
/*                     Original version: R. Ramachandran, 07-Nov-97, nfra.  */
/*                                                                          */
/*  outbuf[]       : input array (short int), replaced by dedispersed data  */
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

void channels_to_disk(unsigned char *subint, char *scratchpath, long int nchans, long int totsize, long long int chanbytes)
{

char tempfilname[250];
long int i,j,k;
FILE** outputfiles = malloc(sizeof(FILE*) * nchans);

for(i = 0; i < nchans; i++) {
   sprintf(tempfilname, "%s/guppi%ld.bin",scratchpath,i);

   outputfiles[i] = fopen(tempfilname, "a+");
   //printf("opening file %s...\n", tempfilname);
}
for(i=0; i<nchans;i++){
	//printf("writing files...%d\n", totsize/nchans);
	fwrite(subint + (i * totsize/nchans), sizeof(char), chanbytes, outputfiles[i]);
}


for(i = 0; i < nchans; i++) {
   fclose(outputfiles[i]);
}

free(outputfiles);
}

int readbin(long int m, unsigned char *channelbuffer, long long int channelbuffer_pos, char *scratchpath){
	 FILE *tempfil;
	 char tempfilname[250];
	 long int i,j,k;
	 
	 sprintf(tempfilname, "%s/guppi%ld.bin",scratchpath,m);
	 if(!exists(tempfilname)) return -1;
	 tempfil = fopen(tempfilname, "r");
	 if(fread(channelbuffer, sizeof(char), channelbuffer_pos, tempfil) != channelbuffer_pos){
		 fclose(tempfil);	
		 return -1;
	 }
	 fclose(tempfil);
	 return 1;
}

int comp_stats(double *mean, double *stddev, float *vec, long int veclen, long int ignore){

	//compute mean and stddev of floating point vector vec, ignoring ignore elements on either side of the vector
	long int i,j,k;
	double tmean = 0;
	double tstddev = 0;
	
	
	for(i=ignore;i < ( veclen - ignore); i++){
		tmean = tmean + vec[i];
	}

	tmean = tmean / (veclen - (2*ignore));
	
	for(i=ignore;i<(veclen-ignore);i++){
		tstddev = tstddev + pow((vec[i] - tmean), 2);
	}
	
	tstddev = pow( (1/ ((double) (veclen - (ignore*2) - 1)) * tstddev), 0.5);
	
	*mean = tmean;
	*stddev = tstddev;
	
	return 0;
}


double chan_freq(double center_freq, double channel_bw, double band_size, long long int fftlen, long int coarse_channel, long int fine_channel, long int tdwidth) {

	double adj_channel_bw = channel_bw + (((tdwidth-fftlen) / fftlen) * channel_bw);
	double adj_fftlen = tdwidth;
	double chanfreq = 0;
	
	chanfreq = (center_freq - (band_size/2)) + (((double) fine_channel * adj_channel_bw)/((double) adj_fftlen)) + ((double) coarse_channel * channel_bw);
	return chanfreq;

}


void simple_fits_write (FILE *fp, float *vec, int tsteps_valid, int freq_channels)
{

	char buf[2880];
	long int bytes_written=0;
	long int i;
	/* zero out header */
	memset(buf, 0x0, 2880);

	
	printf("in func\n");
	strcpy (buf, "END ");
	for(i=4;i<2880;i++) buf[i] = ' ';

	hlength (buf, 2880);
	hadd(buf, "NAXIS2");
	hadd(buf, "NAXIS1");					 
	hadd(buf, "NAXIS");
	hadd(buf, "BITPIX");
	hadd(buf, "SIMPLE");
	hputc(buf, "SIMPLE", "T");
	hputi4(buf, "BITPIX", -32);
	hputi4(buf, "NAXIS", 2);
	hputi4(buf, "NAXIS1", freq_channels);
	hputi4(buf, "NAXIS2", tsteps_valid);
	
		printf("set keywords %d\n",bytes_written);  //write header

	bytes_written = bytes_written + fwrite(buf, sizeof(char), 2880, fp);
	printf("wrote: %d\n",bytes_written);  //write header

	imswap4(vec,(tsteps_valid * freq_channels) * 4);

	bytes_written = bytes_written + fwrite(vec, sizeof(float), tsteps_valid * freq_channels, fp);
	printf("wrote: %d\n",bytes_written);  //write header

	/* create zero pad buffer */
	memset(buf, 0x0, 2880);
	for(i=4;i<2880;i++) buf[i] = ' ';


	bytes_written = bytes_written + fwrite(buf, sizeof(char), 2880 - ((tsteps_valid * freq_channels)%2880), fp);

	printf("wrote: %d\n",bytes_written);  //write header


}



/* IMSWAP4 -- Reverse bytes of Integer*4 or Real*4 vector in place */
void imswap4 (char *string, int nbytes) 
{

/* string Address of Integer*4 or Real*4 vector */
/* bytes Number of bytes to reverse */
    char *sbyte, *slast;
    char temp0, temp1, temp2, temp3;
    slast = string + nbytes;
    sbyte = string;
    while (sbyte < slast) {
        temp3 = sbyte[0];
        temp2 = sbyte[1];
        temp1 = sbyte[2];
        temp0 = sbyte[3];
        sbyte[0] = temp0;
        sbyte[1] = temp1;
        sbyte[2] = temp2;
        sbyte[3] = temp3;
        sbyte = sbyte + 4;
        }

    return;
}
