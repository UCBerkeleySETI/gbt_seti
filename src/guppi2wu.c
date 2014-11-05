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
#include <sys/stat.h>
#include "cufft.h"
#include "guppi2spectra_gpu.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <pthread.h>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



/* Guppi channel-frequency mapping */
/* sample at 1600 MSamp for an 800 MHz BW */
/* N = 256 channels with frequency centers at m * fs/N */
/* m = 0,1,2,3,... 255 */
//#include "filterbank.h"


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
	double baryv;
	double barya;
	unsigned int sqlid;
};

struct max_vals {
	float *maxsnr;
	float *maxdrift;
	unsigned char *maxsmooth;
	unsigned long int *maxid;
};


		
long int simple_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);
		
long int extension_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);

/* prototypes */

float log2f(float x);
double log2(double x);
long int lrint(double x);



extern void explode_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);
extern void detect_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
extern void setQuant(float *lut);
extern void normalize_wrapper(float * tree_dedopplerd_pntr, float *mean, float *stddev, int tdwidth);
extern void vecdivide_wrapper(float * spectrumd, float * divisord, int tdwidth);


struct gpu_spectrometer {
	 unsigned char *channelbuffer;
	 long int channelbuffer_pos;
	 char *scratchpath;
	 cufftComplex *a_h; 
	 cufftComplex *a_d; 
	 cufftHandle plan; 
	 int cufftN; 
	 int cufftbatchSize; 
	 int nBytes;
	 unsigned char *channelbufferd;
	 float * spectrumd;
	 float * spectra;
	 float * bandpassd;
	 struct gpu_input * rawinput;
	 unsigned int gpudevice;
};



int exists(const char *fname);

void print_usage(char *argv[]); 

//int channelize(struct diskinput *diskfiles);

void channels_to_disk(unsigned char *subint, struct gpu_spectrometer *gpu_spec, long int nchans, long int totsize, long long int chanbytes);


double chan_freq(struct gpu_input *firstinput, long long int fftlen, long int coarse_channel, long int fine_channel, long int tdwidth, int ref_frame);

void imswap4 (char *string, int nbytes);

void simple_fits_write (FILE *fp, float *vec, int tsteps_valid, int freq_channels, double fcntr, double deltaf, double deltat, struct guppi_params *g, struct psrfits *p, double snr, double doppler);


void *readbin(void * arg);


/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);



int overlap; // amount of overlap between sub integrations in samples

int N = 128;






int main(int argc, char *argv[]) {


	int filecnt=0;
    char buf[32768];
	

	unsigned char *channelbuffer=NULL;
	float *spectra=NULL;
 	float *spectrum=NULL;
 	struct gpu_spectrometer gpu_spec;
 	
	
	
	char filname[250];

	struct gpu_input rawinput;	
	struct gpu_input firstinput;
	struct max_vals max;
	
	max.maxsnr = NULL;
	max.maxdrift = NULL;
	max.maxsmooth = NULL;
	max.maxid = NULL;
	
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


	int firsttime=0;


	FILE *bandpass_file = NULL;
	float *bandpass;
    
	size_t rv=0;
	long unsigned int by=0;
    
    
	int c;
	long int i,j,k,m,n;
    int vflag=0; //verbose
    
    char *partfilename=NULL;
    char *scratchpath=NULL;

    char *bandpass_file_name=NULL;

    
	rawinput.file_prefix = NULL;
	rawinput.fil = NULL;
	rawinput.invalid = 0;
	rawinput.first_file_skip = 0;

	/* set default gpu device */
	gpu_spec.gpudevice = 0;
	
	long int fftlen;
	fftlen = 32768;

	/* if writing to disk == 1 */
	char disk = 0;
	
    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "Vvdi:o:c:f:b:s:p:g:")) != -1)
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
           case 's':
             scratchpath = optarg;
             break;
           case 'g':
             gpu_spec.gpudevice = atoi(optarg);
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'b':
			 bandpass_file_name = optarg;
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

if(getenv("SETI_GBT") == NULL){
	fprintf(stderr, "Error! SETI_GBT not defined!\n");
	exit(0);
}

if (cudaSetDevice(gpu_spec.gpudevice) != cudaSuccess){
        fprintf(stderr, "Couldn't set GPU device %d\n", gpu_spec.gpudevice);
        exit(0);
}

HANDLE_ERROR ( cudaThreadSynchronize() );



/* load lookup table into GPU memory */

float lookup[4];
 lookup[0] = 3.3358750;
 lookup[1] = 1.0;
 lookup[2] = -1.0;
 lookup[3] = -3.3358750;

setQuant(lookup);

HANDLE_ERROR ( cudaThreadSynchronize() );
/* --------------------------------- */




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


fprintf(stderr, "Will channelize spectra of length %ld pts\n", fftlen);
fprintf(stderr, "outputing detected spectra to %s\n", partfilename);


/* open the first file for input */
sprintf(filname, "%s.0000.raw", rawinput.file_prefix);
rawinput.fil = fopen(filname, "rb");

/* if we managed to open a file */
if(rawinput.fil){
	  if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768){
		   
		   
		   guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);
		   
		   if(rawinput.pf.hdr.nbits == 8) {
			  
			  fprintf(stderr, "caught an an 8 bit header\n");
			  
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

		  fprintf(stderr, "overlap %d\n", rawinput.overlap);
		   fprintf(stderr, "packetindex %lld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
		   fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
		   fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
		   fprintf(stderr, "bytes_per_subint: %d\n\n",rawinput.pf.sub.bytes_per_subint);
		   fprintf(stderr, "nchan: %d\n\n",rawinput.pf.hdr.nchan);
		   fprintf(stderr, "npol: %d\n\n",rawinput.pf.hdr.npol);

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


firstinput = rawinput;


//	tstart=band[first_good_band].pf.hdr.MJD_epoch;

//	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
//	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));

	


printf("calculating index step\n");

/* number of packets that we *should* increment by */
indxstep = (int) ((rawinput.pf.sub.bytes_per_subint * 4) / rawinput.gf.packetsize) - (int) (rawinput.overlap * rawinput.pf.hdr.nchan * rawinput.pf.hdr.rcvr_polns * 2 / rawinput.gf.packetsize);



nchans = rawinput.pf.hdr.nchan;
overlap = rawinput.overlap;

/* number of non-overlapping bytes in each channel */
/* indxstep increments by the number of unique packets in each sub-integration */
/* packetsize is computed based on the original 8 bit resolution */
/* divide by nchan to get to bytes/channel */
/* divide by 4 to get back to 2 bits */


chanbytes = indxstep * rawinput.gf.packetsize / (4 * rawinput.pf.hdr.nchan); 
printf("chan bytes %lld\n", chanbytes);

if(chanbytes%fftlen != 0) {
fprintf(stderr, "samples per channel %lld is not evenly divisible by fftlen %ld!\n", chanbytes, fftlen);
exit(1);
}


channelbuffer  = (unsigned char *) calloc(  indxstep * rawinput.gf.packetsize / (4), sizeof(unsigned char) );
printf("malloc'ing %d Mbytes for processing all channels/all pols\n",  indxstep * rawinput.gf.packetsize / (4));	

/* total number of bytes per channel, including overlap */
chanbytes_overlap = rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan;


printf("Index step: %d\n", indxstep);
printf("bytes per subint %d\n",rawinput.pf.sub.bytes_per_subint );


fflush(stdout);

firsttime=1;

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

if(firsttime) {


	  gpu_spec.cufftN = (int) fftlen;


	  // we'll need to set this to npols * nsamples * nchans / fftsize 
	  gpu_spec.cufftbatchSize = indxstep * rawinput.gf.packetsize / (2 * fftlen);

	
	  gpu_spec.nBytes = sizeof(cufftComplex)*gpu_spec.cufftN*gpu_spec.cufftbatchSize; 

	  bandpass = (float *) malloc (fftlen * sizeof(float));
	  spectra = (float *) malloc (gpu_spec.cufftbatchSize * fftlen * sizeof(float));

	  bandpass_file = fopen(bandpass_file_name, "rb");
	  fread(bandpass, sizeof(float),fftlen, bandpass_file);
	  fclose(bandpass_file);
	  if(vflag>=1) fprintf(stderr,"creating plan for %d point fft with batch size %d\n", gpu_spec.cufftN, gpu_spec.cufftbatchSize);

	  HANDLE_ERROR( cufftPlan1d(&(gpu_spec.plan), gpu_spec.cufftN, CUFFT_C2C, gpu_spec.cufftbatchSize) );

	  //gpu_spec.a_h = (cufftComplex *)malloc(gpu_spec.nBytes);
	  HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec.a_d), gpu_spec.nBytes) ); 
	  HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec.channelbufferd),  indxstep * rawinput.gf.packetsize / 4) );  
	  HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec.bandpassd), fftlen * sizeof(float)) );
	  HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec.spectrumd), gpu_spec.cufftbatchSize * fftlen * sizeof(float)) );

	  HANDLE_ERROR( cudaMemcpy(gpu_spec.bandpassd, bandpass, fftlen * sizeof(float), cudaMemcpyHostToDevice) ); 

	  gpu_spec.channelbuffer = channelbuffer;
	  gpu_spec.channelbuffer_pos = 0;
	  gpu_spec.scratchpath = scratchpath;
	  gpu_spec.spectra = spectra;
	  gpu_spec.rawinput = &rawinput;

	  firsttime=0;
}


					if(vflag>=1) {

						 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
						 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
						 fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
						 fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
						 fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
						 fprintf(stderr, "RA: %f\n\n",rawinput.pf.sub.ra);
						 fprintf(stderr, "DEC: %f\n\n",rawinput.pf.sub.dec);
						 fprintf(stderr, "subintoffset %f\n", rawinput.pf.sub.offs);
						 fprintf(stderr, "tsubint %f\n", rawinput.pf.sub.tsubint);


					}
					
			   		if(rawinput.gf.packetindex == curindx) {

						 /* read a subint with correct index, read the data */
						 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
		
						 if((long int)rv == rawinput.pf.sub.bytes_per_subint){
							if(vflag>=1) fprintf(stderr,"read %d bytes from %ld in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
						 	

							/* need to make sure we only grab the non-overlapping piece */
							for(i = 0; i < rawinput.pf.hdr.nchan; i++) {
								memcpy(channelbuffer + (i * chanbytes), rawinput.pf.sub.data + (i * chanbytes_overlap), chanbytes);												
							}
						 	if(vflag>=1) fprintf(stderr, "copied %lld bytes\n", chanbytes * rawinput.pf.hdr.nchan);
						 	if(vflag>=1) fprintf(stderr, "calling channels_to_disk...");
						 	channels_to_disk(channelbuffer, &gpu_spec, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);

						 	if(vflag>=1) fprintf(stderr, "done\n");

						 } else {
						 	 rawinput.fil = NULL;
						 	 rawinput.invalid = 1;
							 fprintf(stderr,"ERR: couldn't read as much as the header said we could... assuming corruption and exiting...\n");
							 exit(1);
						 }
						 
					
					} else if(rawinput.gf.packetindex > curindx) {
						 fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						 /* read a subint with too high an indx, must have dropped a whole subintegration*/
						 /* don't read the subint, but proceed with re-channelizing the previous subints data */
						 /* so that we maintain continuity in time... */
				

						/* channelbuffer *should* still contain the last valid subint */
						 	
						channels_to_disk(rawinput.pf.sub.data, &gpu_spec, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);
	
						/* We'll get the current valid subintegration again during the next time through this loop */

					
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
									 fprintf(stderr,"Read %d more bytes from %ld in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
		 
		 
		 									 
									 /* need to make sure we only grab the non-overlapping piece */
									 for(i = 0; i < rawinput.pf.hdr.nchan; i++) {
										 memcpy(channelbuffer + (i * chanbytes), rawinput.pf.sub.data + (i * chanbytes_overlap), chanbytes);												
									 }
									 if(vflag>=1) fprintf(stderr, "copied %lld bytes\n", chanbytes * rawinput.pf.hdr.nchan);
													
									 channels_to_disk(channelbuffer, &gpu_spec, nchans, rawinput.pf.sub.bytes_per_subint, chanbytes);
								  
								  
								  } else {
								  	 rawinput.fil = NULL;
									 rawinput.invalid = 1;
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

	
//obs_length = (double) floor(channelbuffer_pos/fftlen) * fftlen * rawinput.pf.hdr.dt; // obs length in seconds
//printf("obs length: %g\n", obs_length);


printf("start time %15.15Lg end time %15.15Lg %s %s barycentric velocity %15.15g barycentric acceleration %15.15g \n", firstinput.pf.hdr.MJD_epoch, rawinput.pf.hdr.MJD_epoch, firstinput.pf.hdr.ra_str, firstinput.pf.hdr.dec_str, firstinput.baryv, firstinput.barya);

	





	free(spectrum);

	
	/* free original array */
	free(spectra);	
	
	free(channelbuffer);
	free(bandpass);
	
	
	fprintf(stderr, "cleaned up FFTs...\n");

	
	

	fprintf(stderr, "closed output file...\n");


    exit(1);


}



/*
int channelize(struct diskinput *diskfiles)
{
int i,j,k;


float lookup[4];
lookup[0] = 3.3358750;
lookup[1] = 1.0;
lookup[2] = -1.0;
lookup[3] = -3.3358750;


// number of fft frames in channel buffer 
long long int numframes;
fprintf(stderr, "calculating nframes\n");

numframes = (long long int) floor(diskfiles->channelbuffer_pos/diskfiles->cufftN);

fprintf(stderr, "done\n");

setQuant(lookup);
cudaThreadSynchronize();

fprintf(stderr, "executing fft over %lld frames \n", numframes);

fprintf(stderr, "copying onto gpu: %d\n", cudaMemcpy(diskfiles->channelbufferd, diskfiles->channelbuffer, diskfiles->channelbuffer_pos, cudaMemcpyHostToDevice)); 

		for(i = 0; i < numframes; i++) {
		
			explode_wrapper(diskfiles->channelbufferd + (diskfiles->cufftN * i), diskfiles->a_d, diskfiles->cufftN);
			cudaThreadSynchronize();
		
			cufftExecC2C(diskfiles->plan, diskfiles->a_d, diskfiles->a_d, CUFFT_FORWARD); 
			cudaThreadSynchronize();
			detect_wrapper(diskfiles->a_d, diskfiles->cufftN, diskfiles->bandpassd, diskfiles->spectrumd);
			cudaThreadSynchronize();
		
			cudaMemcpy(diskfiles->spectra + (i * diskfiles->cufftN), diskfiles->spectrumd, diskfiles->cufftN * sizeof(float), cudaMemcpyDeviceToHost);
		
		
			// set the DC channel equal to the mean of the two adjacent channels
			diskfiles->spectra[ (diskfiles->cufftN/2) + (i * diskfiles->cufftN) ] = (diskfiles->spectra[ (diskfiles->cufftN/2) + (i * diskfiles->cufftN) - 1 ] + diskfiles->spectra[ (diskfiles->cufftN/2) + (i * diskfiles->cufftN) + 1])/2;
			
		}	



return 0;
}

*/

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





void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input_prefix -c channel -p N\n", argv[0]);
	fprintf(stderr, "		N = 2^N FFT Points\n");
	fprintf(stderr, "		-v or -V for verbose\n");
}







void channels_to_disk(unsigned char *subint, struct gpu_spectrometer *gpu_spec, long int nchans, long int totsize, long long int chanbytes)
{

char tempfilname[250];
long int i,j,k;
FILE * outputfile;
long int nframes;
long int fitslen;
float * stitched_spectrum;
char *fitsdata;
/* chan 0, pol 0, r, pol 0 i, pol 1 ... */

		   //fprintf(stderr, "center_freq: %f\n\n", gpu_spec->rawinput->pf.hdr.fctr);
	nframes = chanbytes / gpu_spec->cufftN;

	
	   //fprintf(stderr, "0x%08x 0x%08x", subint[100], subint[1]);

	   //fprintf(stderr, "%f\n", tframe);
	   cudaThreadSynchronize();

	   /* copy whole subint onto gpu */
	   HANDLE_ERROR( cudaMemcpy( gpu_spec->channelbufferd, subint, (size_t) chanbytes * nchans, cudaMemcpyHostToDevice) ); 
	   
	   /* explode to a floating point array twice the length of nsamples, one for each polarization */
	   explode_wrapper(gpu_spec->channelbufferd, gpu_spec->a_d, chanbytes * nchans);
	   cudaThreadSynchronize();
	   
	   HANDLE_ERROR( cufftExecC2C(gpu_spec->plan, gpu_spec->a_d, gpu_spec->a_d, CUFFT_FORWARD) ); 
	   detect_wrapper(gpu_spec->a_d, chanbytes * nchans, gpu_spec->cufftN, gpu_spec->bandpassd, gpu_spec->spectrumd);
	   
	   cudaThreadSynchronize();
	   HANDLE_ERROR( cudaMemcpy(gpu_spec->spectra, gpu_spec->spectrumd, chanbytes * nchans * sizeof(float), cudaMemcpyDeviceToHost) );
 	    //fprintf(stderr, "\n\n%f %f\n\n", gpu_spec->spectra[100], gpu_spec->spectra[1]);
	//	exit(0);
	   	   
	   //for(k = 0; k < nframes; k++) {
		//	for(j = 0; j < gpu_spec->cufftN; j++) {
		//		stitched_spectrum[ (gpu_spec->cufftN * i) + j ] = stitched_spectrum[ (gpu_spec->cufftN * i) + j ] + gpu_spec->spectra[(k * gpu_spec->cufftN) + ( (j+gpu_spec->cufftN/2)%gpu_spec->cufftN )];				 
		//	}
	   //}
	  // set the DC channel equal to the mean of the two adjacent channels
	   //stitched_spectrum[ (gpu_spec->cufftN/2) + (i * gpu_spec->cufftN) ] = (stitched_spectrum[ (gpu_spec->cufftN/2) + (i * gpu_spec->cufftN) - 1 ] + stitched_spectrum[ (gpu_spec->cufftN/2) + (i * gpu_spec->cufftN) + 1])/2;

//meta information:
//RA (double), DEC (double), MJD (double), Center frequency (double)
sprintf(tempfilname, "%s/%s.%f.fits",gpu_spec->scratchpath, gpu_spec->rawinput->pf.hdr.source, gpu_spec->rawinput->pf.hdr.fctr);
outputfile = fopen(tempfilname, "a+");
//for(i=0;i<chanbytes * nchans;i++) {
//fprintf(outputfile, "%f\n", gpu_spec->spectra[i]);
//}
fitsdata = malloc((chanbytes * nchans * sizeof(float)) + 2880 * 2);

if(gpu_spec->channelbuffer_pos == 0) {
	 gpu_spec->rawinput->pf.sub.offs = (double) nchans / (double) (gpu_spec->rawinput->pf.hdr.BW * 1000000) * (double) chanbytes * (double) (gpu_spec->channelbuffer_pos + 0.5);
	 fitslen = simple_fits_buf(fitsdata, (float *) gpu_spec->spectra, 1, nchans*chanbytes, gpu_spec->rawinput->pf.hdr.fctr, (long int) gpu_spec->cufftN, 0.0, 0.0, gpu_spec->rawinput);
} else {
 	 gpu_spec->rawinput->pf.sub.offs = (double) nchans / (double) (gpu_spec->rawinput->pf.hdr.BW * 1000000) * (double) chanbytes * (double) (gpu_spec->channelbuffer_pos);
	 fitslen = extension_fits_buf(fitsdata, (float *) gpu_spec->spectra, 1, nchans*chanbytes, gpu_spec->rawinput->pf.hdr.fctr, (long int) gpu_spec->cufftN, 0.0, 0.0, gpu_spec->rawinput);
}

fwrite(fitsdata, sizeof(char), fitslen, outputfile);
fflush(outputfile);
fclose(outputfile);
free(fitsdata);

//fwrite(gpu_spec->spectra, sizeof(char), chanbytes * nchans * sizeof(float), outputfile);


    gpu_spec->channelbuffer_pos = gpu_spec->channelbuffer_pos + 1;

}



double chan_freq(struct gpu_input *firstinput, long long int fftlen, long int coarse_channel, long int fine_channel, long int tdwidth, int ref_frame) {

	double center_freq = firstinput->pf.hdr.fctr;
	double channel_bw = firstinput->pf.hdr.df;
	double band_size = firstinput->pf.hdr.BW;
	double adj_channel_bw = channel_bw + (((tdwidth-fftlen) / fftlen) * channel_bw);
	double adj_fftlen = tdwidth;
	double chanfreq = 0;
	double bandpad = (((tdwidth-fftlen) / fftlen) * channel_bw);
	//chan_freq = center_freq - (band_size/2) + ((double) coarse_channel * channel_bw) - bandpad

	/* determing channel frequency from fine and coarse bins */
	chanfreq = (center_freq - (band_size/2)) + (((double) fine_channel * adj_channel_bw)/((double) adj_fftlen)) + ((double) coarse_channel * channel_bw) - (bandpad/2);

	/* apply doppler correction */
	if(ref_frame == 1) chanfreq = (1 - firstinput->baryv) * chanfreq;

	return chanfreq;

}

long int extension_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput)
{

char * buf;
long int i,j,k;
long int fitslen=0;
buf = (char *) malloc(2880 * sizeof(char));
/* zero out header */
memset(buf, 0x0, 2880);

	strcpy (buf, "END ");
	for(i=4;i<2880;i++) buf[i] = ' ';

	hlength (buf, 2880);



	hadd(buf, "SOURCE");
	hadd(buf, "SNR");
	hadd(buf, "DOPPLER");
	hadd(buf, "RA");
	hadd(buf, "DEC");
	hadd(buf, "TOFFSET");
	hadd(buf, "FCNTR");
	hadd(buf, "DELTAF");
	hadd(buf, "DELTAT");
	hadd(buf, "GCOUNT");
	hadd(buf, "PCOUNT");
	hadd(buf, "NAXIS2");
	hadd(buf, "NAXIS1");
	hadd(buf, "NAXIS");
	hadd(buf, "BITPIX");
	hadd(buf, "XTENSION");


	hputs(buf, "XTENSION", "IMAGE");
	hputi4(buf, "BITPIX", -32);
	hputi4(buf, "NAXIS", 2);
	hputi4(buf, "NAXIS1", width);
	hputi4(buf, "NAXIS2", height);
	hputi4(buf, "GCOUNT", 1);
	hputi4(buf, "PCOUNT", 1);

	hputnr8(buf, "FCNTR", 12, fcntr);
	hputnr8(buf, "DELTAF", 12, (double) firstinput->pf.hdr.df/fftlen);
	hputnr8(buf, "DELTAT", 12, (double) fftlen/(1000000 * firstinput->pf.hdr.df));

	hputnr8(buf, "TOFFSET", 12, (double) (firstinput->pf.sub.offs) );
	hputnr8(buf, "RA", 12, firstinput->pf.sub.ra);
	hputnr8(buf, "DEC", 12, firstinput->pf.sub.dec);
	hputnr8(buf, "DOPPLER", 12, doppler);
	hputnr8(buf, "SNR", 12, snr);
	hputs(buf, "SOURCE", firstinput->pf.hdr.source);

	memcpy(fitsdata, buf, 2880 * sizeof(char));
	fitslen = fitslen + 2880;
	imswap4((char *) vec,(height * width) * 4);
	
	memcpy(fitsdata+2880, vec, (height * width) * 4);
	fitslen = fitslen + (height * width * 4);
	/* create zero pad buffer */
	memset(buf, 0x0, 2880);
	for(i=0;i<2880;i++) buf[i] = ' ';
		
	memcpy(fitsdata + 2880 + (height * width * 4), buf, 2880 - ((height*width*4)%2880));
	fitslen = fitslen + 2880 - ((height*width*4)%2880);
	free(buf);
	return fitslen;
}



long int simple_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput)
{

char * buf;
long int i,j,k;
long int fitslen=0;
buf = (char *) malloc(2880 * sizeof(char));
/* zero out header */
memset(buf, 0x0, 2880);

	strcpy (buf, "END ");
	for(i=4;i<2880;i++) buf[i] = ' ';

	hlength (buf, 2880);			 


	hadd(buf, "SOURCE");
	hadd(buf, "SNR");
	hadd(buf, "DOPPLER");
	hadd(buf, "RA");
	hadd(buf, "DEC");
	hadd(buf, "MJD");
	hadd(buf, "FCNTR");
	hadd(buf, "DELTAF");
	hadd(buf, "DELTAT");
	hadd(buf, "EXTEND");
	hadd(buf, "NAXIS2");
	hadd(buf, "NAXIS1");
	hadd(buf, "NAXIS");
	hadd(buf, "BITPIX");
	hadd(buf, "SIMPLE");
		
	hputc(buf, "SIMPLE", "T");
	hputi4(buf, "BITPIX", -32);
	hputi4(buf, "NAXIS", 2);
	hputi4(buf, "NAXIS1", width);
	hputi4(buf, "NAXIS2", height);
	hputc(buf, "EXTEND", "T");

	hputnr8(buf, "FCNTR", 12, fcntr);
	hputnr8(buf, "DELTAF", 12, (double) firstinput->pf.hdr.df/fftlen);
	hputnr8(buf, "DELTAT", 12, (double) fftlen/(1000000 * firstinput->pf.hdr.df));

	hputnr8(buf, "MJD", 12, (double) firstinput->pf.hdr.MJD_epoch + (double) (firstinput->pf.sub.offs/86400.00) );
	hputnr8(buf, "RA", 12, firstinput->pf.sub.ra);
	hputnr8(buf, "DEC", 12, firstinput->pf.sub.dec);
	hputnr8(buf, "DOPPLER", 12, doppler);
	hputnr8(buf, "SNR", 12, snr);
	hputs(buf, "SOURCE", firstinput->pf.hdr.source);

	memcpy(fitsdata, buf, 2880 * sizeof(char));
	fitslen = fitslen + 2880;
	imswap4((char *) vec,(height * width) * 4);
	
	memcpy(fitsdata+2880, vec, (height * width) * 4);
	fitslen = fitslen + (height * width * 4);
	/* create zero pad buffer */
	memset(buf, 0x0, 2880);
	for(i=0;i<2880;i++) buf[i] = ' ';
		
	memcpy(fitsdata + 2880 + (height * width * 4), buf, 2880 - ((height*width*4)%2880));
	fitslen = fitslen + 2880 - ((height*width*4)%2880);
	free(buf);
	return fitslen;
}



void simple_fits_write (FILE *fp, float *vec, int tsteps_valid, int freq_channels, double fcntr, double deltaf, double deltat, struct guppi_params *g, struct psrfits *p, double snr, double doppler)
{

	char buf[2880];
	long int bytes_written=0;
	long int i;
	/* zero out header */
	memset(buf, 0x0, 2880);

	
	strcpy (buf, "END ");
	for(i=4;i<2880;i++) buf[i] = ' ';

	hlength (buf, 2880);

	
	hadd(buf, "SOURCE");
	hadd(buf, "SNR");
	hadd(buf, "DOPPLER");
	hadd(buf, "RA");
	hadd(buf, "DEC");
	hadd(buf, "MJD");
	hadd(buf, "FCNTR");
	hadd(buf, "DELTAF");
	hadd(buf, "DELTAT");
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
	hputnr8(buf, "FCNTR", 12, fcntr);
	hputnr8(buf, "DELTAF", 12, deltaf);
	hputnr8(buf, "DELTAT", 12, deltat);

	hputnr8(buf, "MJD", 12, (double) p->hdr.MJD_epoch);
	hputnr8(buf, "RA", 12, p->sub.ra);
	hputnr8(buf, "DEC", 12, p->sub.dec);
	hputnr8(buf, "DOPPLER", 12, doppler);
	hputnr8(buf, "SNR", 12, snr);
	hputc(buf, "SOURCE", p->hdr.source);
	
		printf("set keywords %ld\n",bytes_written);  //write header

	bytes_written = bytes_written + fwrite(buf, sizeof(char), 2880, fp);
	printf("wrote: %ld\n",bytes_written);  //write header

	imswap4((char *) vec,(tsteps_valid * freq_channels) * 4);

	bytes_written = bytes_written + fwrite(vec, sizeof(float), tsteps_valid * freq_channels, fp);
	printf("wrote: %ld\n",bytes_written);  //write header

	/* create zero pad buffer */
	memset(buf, 0x0, 2880);
	for(i=4;i<2880;i++) buf[i] = ' ';


	bytes_written = bytes_written + fwrite(buf, sizeof(char), 2880 - ((tsteps_valid * freq_channels*4)%2880), fp);

	printf("wrote: %ld\n",bytes_written);  //write header


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
