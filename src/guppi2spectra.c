#define _GLIBCXX_USE_CXX11_ABI 0
#define MAXSIZE 134000000

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
#include "filterbank.h"
#include "fcntl.h"
#include "imswap.h"


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

static void HandleCufftError( cufftResult err,
                         const char *file,
                         int line ) {
    if (err != CUFFT_SUCCESS) {
        printf( "%s in %s at line %d\n", _cudaGetCufftErrorEnum( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUFFT_ERROR( err ) (HandleCufftError( err, __FILE__, __LINE__ ))


// Instead of copying the "strings_equal" function from one of several source
// files in this directory(!), just define it as a macro.
#define strings_equal(a,b) (!strcmp(a,b))


struct gpu_input {
	char *file_prefix;
	struct guppi_params gf;
	struct psrfits pf;	
	unsigned int filecnt;
	FILE *fil;
	FILE *headerfile;
	int invalid;
	int curfile;
	int overlap;   /* add this keyword here since it doesn't seem to appear in guppi_params.c */
	long int first_file_skip; /* in case there's 8bit data in the header of file 0 */
	double baryv;
	double barya;
	unsigned int sqlid;
	long int nsamples;
	int indxstep;
	int chanbytes;
	long int num_bufs;

};



		
long int simple_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);
		
long int extension_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);

/* prototypes */

float log2f(float x);
double log2(double x);
long int lrint(double x);



/* Wrapper functions for performing various spectroscopy options on the GPU */
#ifdef __cplusplus
extern "C" {
#endif
void explode_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);
void detect_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
void detectX_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
void detectY_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
void detectV_wrapper(cufftComplex * voltages, int veclen, int fftlen, float *bandpassd, float *spectrumd);
void setQuant(float *lut);
void setQuant8(float *lut);
void normalize_wrapper(float * tree_dedopplerd_pntr, float *mean, float *stddev, int tdwidth);
void vecdivide_wrapper(float * spectrumd, float * divisord, int tdwidth);
void explode8_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
void explode8init_wrapper(char *channelbufferd, long int length);
void explode8simple_wrapper(char *channelbufferd, cufftComplex * voltages, int veclen);
void explode8lut_wrapper(unsigned char *channelbufferd, cufftComplex * voltages, int veclen);
#ifdef __cplusplus
}
#endif

/* gpu spectrometer structure */
/* one of these for each cufft plan that will operate on a block of raw voltage data */
/* 'd' elements are on the GPU (device) */

struct gpu_spectrometer {
	 unsigned char *channelbuffer;
	 long int spectracnt;
	 long int triggerwrite;
	 cufftComplex *a_d; 
	 cufftComplex *b_d; 
	 cufftHandle plan; 
	 long int cufftN; 
	 long int cufftbatchSize; 
	 long int integrationtime;
	 long int spectraperchannel;
	 int kurtosis;
	 int pol;
	 unsigned char *channelbufferd;
	 char *channelbufferd8;
	 float * spectrumd;
	 float * spectra;
	 float * spectrum;
	 float * bandpass;
	 float * bandpassd;
	 struct gpu_input * rawinput;
	 unsigned int gpudevice;
	 int nspec;
	 FILE *filterbank_file;
	 char filename[255];
};





int exists(const char *fname);

void print_usage(char *argv[]); 

//int channelize(struct diskinput *diskfiles);

void gpu_channelize(struct gpu_spectrometer gpu_spec[4], long int nchannels, long long int nsamples);

void accumulate_write(void *ptr);

void accumulate_write_single(void *ptr);


double chan_freq(struct gpu_input *firstinput, long long int fftlen, long int coarse_channel, long int fine_channel, long int tdwidth, int ref_frame);


void simple_fits_write (FILE *fp, float *vec, int tsteps_valid, int freq_channels, double fcntr, double deltaf, double deltat, struct guppi_params *g, struct psrfits *p, double snr, double doppler);


void *readbin(void * arg);


/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);



int overlap; // amount of overlap between sub integrations in samples

int N = 128;

int vflag=0; //verbose






int main(int argc, char *argv[]) {


    char buf[32768];
	char tempbuf[16];
	char tempbufl[256];

	
	pthread_t accumwrite_th0;
	pthread_t accumwrite_th1;
	pthread_t accumwrite_th2;
	pthread_t accumwrite_th3;
	

	unsigned char *channelbuffer=NULL;

 	struct gpu_spectrometer gpu_spec[4];
 	
	
	
	char filname[250];

	struct gpu_input rawinput;	
	struct gpu_input firstinput;

	
	
	long long int startindx;
	long long int curindx;
	long long int chanbytes=0;
	long long int chanbytes_overlap = 0;
	long long int nsamples = 0;
	long int hlength=0;
	double hack = 0;
	long int nchannels = 0;
	int inbits = 0;
	int indxstep = 0;
	int channel = -1;
    


    
	size_t rv=0;
	long unsigned int by=0;
    
    
	int c;
	long int i,j,k,m,n;
    
    char *partfilename=NULL;


    
	rawinput.file_prefix = NULL;
	rawinput.fil = NULL;
	rawinput.invalid = 0;
	rawinput.first_file_skip = 0;

	int filehandle = -1;
	
	char *pntr;

	int fftcnt=0, integrationcnt=0;


	   /* set default gpu device */
	   for(i=0;i<4;i++){ gpu_spec[i].gpudevice = 0; }

	   /* set default polarization mode (2 == Stokes I) */
	   for(i=0;i<4;i++){ gpu_spec[i].pol = 2; }
	
    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }

//Necessary inputs:
//number of data bufs to read in to GPU and channelize
//length of FFT per coarse channel
//number of spectra to sum (tscrunch)

long int num_bufs=1;
       opterr = 0;
     
       while ((c = getopt (argc, argv, "Vvdi:o:c:h:f:t:k:b:p:g:B:")) != -1)
         switch (c)
           {
           case 'v':
             vflag = 1;
             break;
           case 'B':
             num_bufs = atoi(optarg);
             break;             
           case 'f':
              i=0;
              pntr = strtok (optarg,",");
			  while (pntr != NULL) {
				  if(i>3){fprintf(stderr, "Error! Up to four FFT lengths supported.\n"); exit(1);}
				  gpu_spec[i].cufftN=atol(pntr);
				  pntr = strtok (NULL, ",");
			  	  i++;
			  }
			  fftcnt = i;
             break;
            case 't':
              i=0;
              pntr = strtok (optarg,",");
			  while (pntr != NULL) {
				  if(i>3){fprintf(stderr, "Error! Up to four integration lengths supported.\n"); exit(1);}
				  gpu_spec[i].integrationtime=atol(pntr);
				  pntr = strtok (NULL, ",");
			  	  i++;
			  }
			  integrationcnt = i;
             break;              
            case 'k':
              i=0;
              pntr = strtok (optarg,",");
			  while (pntr != NULL) {
				  if(i>3){fprintf(stderr, "Error! Up to four kurtosis instances supported.\n"); exit(1);}
				  gpu_spec[i].kurtosis=atoi(pntr);
				  pntr = strtok (NULL, ",");
			  	  i++;
			  }
             break;              
           case 'b':
             inbits = atoi(optarg);
             break;
           case 'p':
           	 for(i=0;i<4;i++){ gpu_spec[i].pol = atoi(optarg); } //0 (X), 1 (Y) or 2 (summed, Stokes-I)
             break;
           case 'g':
           	 for(i=0;i<4;i++){ gpu_spec[i].gpudevice = atoi(optarg); }
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'h':
             hack = strtod(optarg, NULL);
             //1500.0/1024.0;
             fprintf (stderr, "Setting hacked frequency offset to: %f \n", hack);
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

if ((integrationcnt == 0) || (fftcnt == 0)) {fprintf(stderr, "Must specify at least one FFT length and integration time.\n"); exit(1);}
if (integrationcnt != fftcnt) {fprintf(stderr, "Must specify same number of integration times as FFT lengths.\n"); exit(1);}


/* set number of spectrometer instances */
gpu_spec[0].nspec = fftcnt;




if(strstr(rawinput.file_prefix, ".0000.raw") != NULL) memset(rawinput.file_prefix + strlen(rawinput.file_prefix) - 9, 0x0, 9);


//if(getenv("SETI_GBT") == NULL){
//	fprintf(stderr, "Error! SETI_GBT not defined!\n");
//	exit(0);
//}

if (cudaSetDevice(gpu_spec[0].gpudevice) != cudaSuccess){
        fprintf(stderr, "Couldn't set GPU device %d\n", gpu_spec[0].gpudevice);
        exit(0);
}

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






/* open the first file for input */
sprintf(filname, "%s.0000.raw", rawinput.file_prefix);
rawinput.fil = fopen(filname, "rb");

/* if we managed to open a file */
if(rawinput.fil){
	  if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768){
		   
		   
		   guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);

		   if (inbits == 0) {fprintf(stderr, "Warning: input bitwidth not set, defaulting to %d\n", rawinput.pf.hdr.nbits); inbits = rawinput.pf.hdr.nbits;}

		   if(rawinput.pf.hdr.nbits == 8 && inbits == 2) {
			  /* note this should never be the case for directio = 1 data... */
			  fprintf(stderr, "Found an 8 bit header for 2 bit data, skipping...\n");
			  
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
		   fprintf(stderr, "DIRECTIO: %d\n\n",rawinput.pf.hdr.directio);

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


fprintf(stderr, "Running with the following parameters...\n");

fprintf(stderr, "Will load %ld blocks (subints) into GPU\n\n", num_bufs);
fprintf(stderr, "Will channelize in %d modes:\n", gpu_spec[0].nspec);

for (i = 0; i < gpu_spec[0].nspec; i++) {
	fprintf(stderr, "FFT Length: %ld  Integration Time: %ld\n", gpu_spec[i].cufftN,gpu_spec[i].integrationtime);
}
			

fprintf(stderr, "outputing detected spectra to %s\n", partfilename);
fprintf(stderr, "Total Memory Footprint on GPU (2bit): %f MB \n", (double) rawinput.pf.sub.bytes_per_subint * num_bufs / 1048576.0);
fprintf(stderr, "Total Memory Footprint on GPU (32 bit float): %f MB \n", (double) rawinput.pf.sub.bytes_per_subint * num_bufs * 16 / 1048576.0);
fprintf(stderr, "Available samples per channel %f\n", (double) rawinput.pf.sub.bytes_per_subint * num_bufs / rawinput.pf.hdr.nchan);
fprintf(stderr, "nbits: %d\n\n",rawinput.pf.hdr.nbits);



firstinput = rawinput;

	

//	tstart=band[first_good_band].pf.hdr.MJD_epoch;

//	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
//	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));

	


if(vflag>1) fprintf(stderr, "calculating index step\n");

/* number of packets that we *should* increment by */
indxstep = (int) ((rawinput.pf.sub.bytes_per_subint * (8/rawinput.pf.hdr.nbits)) / rawinput.gf.packetsize) - (int) (rawinput.overlap * rawinput.pf.hdr.nchan * rawinput.pf.hdr.rcvr_polns * 2 / rawinput.gf.packetsize);
rawinput.indxstep = indxstep;
rawinput.num_bufs = num_bufs;

nchannels = rawinput.pf.hdr.nchan;

overlap = rawinput.overlap;

/* number of non-overlapping bytes in each channel */
/* indxstep increments by the number of unique packets in each sub-integration */
/* packetsize is computed based on the original 8 bit resolution */
/* divide by nchan to get to bytes/channel */
/* divide by 4 to get back to 2 bits */


chanbytes = (indxstep * rawinput.gf.packetsize / ((8/rawinput.pf.hdr.nbits) * rawinput.pf.hdr.nchan)) * num_bufs; 
rawinput.chanbytes = chanbytes;

if(vflag>1) fprintf(stderr, "number of non-overlapping bytes for each chan %lld\n", chanbytes);

/* for 2 bit data, the number of dual pol samples is the same as the number of bytes */

nsamples = chanbytes / 4 * (8/rawinput.pf.hdr.nbits);
rawinput.nsamples = nsamples;

for(i=0;i<gpu_spec[0].nspec;i++){
   if((nsamples)%gpu_spec[i].cufftN != 0) {
   fprintf(stderr, "samples per channel %lld is not evenly divisible by fftlen %ld!\n", nsamples, gpu_spec[i].cufftN);
   exit(1);
   }
}


channelbuffer  = (unsigned char *) calloc(chanbytes * nchannels, sizeof(unsigned char) );
if(vflag>0)printf("malloc'ing %Ld Mbytes for processing all channels/all pols\n",  chanbytes * nchannels);	

long int chanbytes_subint;
long int chanbytes_subint_total;
chanbytes_subint = (indxstep * rawinput.gf.packetsize / ((8/rawinput.pf.hdr.nbits) * rawinput.pf.hdr.nchan));
chanbytes_subint_total = rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan;


/* total number of bytes per channel, including overlap */
chanbytes_overlap = rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan;
if(vflag>0)fprintf(stderr, "total number of bytes per channel, including overlap %lld\n", chanbytes_overlap);


if(vflag>0)fprintf(stderr, "Index step: %d\n", indxstep);
if(vflag>0)fprintf(stderr, "bytes per subint %d\n",rawinput.pf.sub.bytes_per_subint );


fflush(stdout);


startindx = rawinput.gf.packetindex;
curindx = startindx;

rawinput.curfile = 0;			
long int subint_cnt = 0;


 /* First setup all the gpu stuff... */

 size_t totalsize, worksize;

 /* Handle everything that's uniform between spectrometer instances */
HANDLE_ERROR( cudaMemGetInfo(&worksize, &totalsize));
if(vflag>0) fprintf(stderr,"Memory Free: %ld  Total Memory: %ld\n", (long int) (((double) worksize) /  1048576.0), (long int) (((double) totalsize) /  1048576.0) );


HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[0].a_d), sizeof(cufftComplex)*(nsamples * nchannels * 2)) ); 


/* load lookup table into GPU memory */

if (rawinput.pf.hdr.nbits == 2) {
	if(vflag>0) fprintf(stderr,"Initializing for 2 bits... chanbytes: %lld nchannels: %ld\n", chanbytes, nchannels);

	float lookup[4];
	 lookup[0] = 3.3358750;
	 lookup[1] = 1.0;
	 lookup[2] = -1.0;
	 lookup[3] = -3.3358750;
	setQuant(lookup);
	HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[0].channelbufferd),  chanbytes * nchannels) );  	

} else if (rawinput.pf.hdr.nbits == 8) {
	if(vflag>0) fprintf(stderr,"Initializing for 8 bits... chanbytes: %lld nchannels: %ld\n", chanbytes, nchannels);


	//HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[0].channelbufferd8),  chanbytes * nchannels) );  	
	//HANDLE_ERROR ( cudaThreadSynchronize() );
	//explode8init_wrapper(gpu_spec[0].channelbufferd8, chanbytes * nchannels);

	/* try unsigned for lookup table... */
	float lookup[256];
	for(i=0;i<128;i++) lookup[i] = (float) i;
	for(i=128;i<256;i++) lookup[i] = (float) (i - 256);
	setQuant8(lookup);
	HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[0].channelbufferd),  chanbytes * nchannels) );  	

}

HANDLE_ERROR ( cudaThreadSynchronize() );

for(i=1;i<gpu_spec[0].nspec;i++){
	gpu_spec[i].channelbufferd = gpu_spec[0].channelbufferd;
	gpu_spec[i].channelbufferd8 = gpu_spec[0].channelbufferd8;
	gpu_spec[i].a_d = gpu_spec[0].a_d;
}

for(i=0;i<gpu_spec[0].nspec;i++){
	gpu_spec[i].cufftbatchSize = (nsamples * nchannels * 2)/gpu_spec[i].cufftN;
	gpu_spec[i].spectraperchannel = (long int) (nsamples)/gpu_spec[i].cufftN;



	if( gpu_spec[i].spectraperchannel > gpu_spec[i].integrationtime) {
		if (gpu_spec[i].spectraperchannel%gpu_spec[i].integrationtime != 0) {
			fprintf(stderr,"%ld not evenly divisible by %ld!\n", gpu_spec[i].spectraperchannel, gpu_spec[i].integrationtime);        		
			exit(1);
		}
	} else if( gpu_spec[i].spectraperchannel < gpu_spec[i].integrationtime) {
		if (gpu_spec[i].integrationtime%gpu_spec[i].spectraperchannel != 0) {
			fprintf(stderr,"%ld not evenly divisible by %ld!\n", gpu_spec[i].integrationtime, gpu_spec[i].spectraperchannel);        		
			exit(1);
		}
	}        

	gpu_spec[i].bandpass = (float *) malloc (gpu_spec[i].cufftN * sizeof(float));
	gpu_spec[i].spectra = (float *) malloc (gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN * sizeof(float));
	gpu_spec[i].spectrum = (float *) malloc (gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN * sizeof(float));
	
	if(vflag>=1) fprintf(stderr,"Spectra per channel for this mode: %ld\n", gpu_spec[i].spectraperchannel);        		
    if(vflag>=1) fprintf(stderr,"Memory Free: %ld  Total Memory: %ld\n", (long int) (((double) worksize) /  1048576.0), (long int) (((double) totalsize) /  1048576.0) );

	if ((gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN) > MAXSIZE) {
		if(vflag>=1) fprintf(stderr,"Planning FFT Size %ld Batch Size: %ld (SPLIT)\n", gpu_spec[i].cufftN, gpu_spec[i].cufftbatchSize/2);
		HANDLE_CUFFT_ERROR( cufftPlan1d(&(gpu_spec[i].plan), gpu_spec[i].cufftN, CUFFT_C2C, gpu_spec[i].cufftbatchSize/2) );
		if(vflag>=1) HANDLE_CUFFT_ERROR( cufftGetSize1d(gpu_spec[i].plan, gpu_spec[i].cufftN, CUFFT_C2C, gpu_spec[i].cufftbatchSize/2, &worksize) );
	} else {
		if(vflag>=1) fprintf(stderr,"Planning FFT Size %ld Batch Size: %ld\n", gpu_spec[i].cufftN, gpu_spec[i].cufftbatchSize);
		HANDLE_CUFFT_ERROR( cufftPlan1d(&(gpu_spec[i].plan), gpu_spec[i].cufftN, CUFFT_C2C, gpu_spec[i].cufftbatchSize) );
		if(vflag>=1) HANDLE_CUFFT_ERROR( cufftGetSize1d(gpu_spec[i].plan, gpu_spec[i].cufftN, CUFFT_C2C, gpu_spec[i].cufftbatchSize, &worksize) );
	}
	
	if(vflag>=1) fprintf(stderr,"Estimated Size of Work Space: %ld\n", (long int) (((double) worksize) /  1048576.0) );
	HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[i].bandpassd), gpu_spec[i].cufftN * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[i].spectrumd), gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void **)&(gpu_spec[i].b_d), sizeof(cufftComplex)*(nsamples * nchannels * 2)) ); 
	HANDLE_ERROR( cudaMemcpy(gpu_spec[i].bandpassd, gpu_spec[i].bandpass, gpu_spec[i].cufftN * sizeof(float), cudaMemcpyHostToDevice) ); 

	gpu_spec[i].channelbuffer = channelbuffer;
	gpu_spec[i].spectracnt = 0;
	gpu_spec[i].triggerwrite = 0;
	
	gpu_spec[i].rawinput = &rawinput;
	for(j=0;j<(nchannels * gpu_spec[i].cufftN);j++) gpu_spec[i].spectrum[j] = 0;
	for(j=0;j<gpu_spec[i].cufftN;j++)  gpu_spec[i].bandpass[j] = 1;
}

cudaThreadSynchronize();

HANDLE_ERROR( cudaMemGetInfo(&worksize, &totalsize));
if(vflag>=1) fprintf(stderr,"Done: Memory Free: %ld  Total Memory: %ld\n", (long int) (((double) worksize) /  1048576.0), (long int) (((double) totalsize) /  1048576.0) );

/* Now let's do the filterbank headers... */

machine_id=20;
telescope_id=6;
data_type=1;
nbeams = 1;
ibeam = 1;
nbits=32;
obits=32;
nifs = 1;
src_raj=0.0;
src_dej=0.0;
az_start=0.0;
za_start=0.0;
strcpy(ifstream,"YYYY");
tstart = rawinput.pf.hdr.MJD_epoch;



sprintf(source_name, "%s", rawinput.pf.hdr.source);
sprintf(inpfile, "%s", filname);
memset(tempbufl, 0x0, 256);
strcat(tempbufl, strtok(rawinput.pf.hdr.ra_str, ":"));

strcat(tempbufl, strtok((char *) 0, ":"));
strcat(tempbufl, strtok((char *) 0, ":"));
src_raj = strtod(tempbufl, (char **) NULL);

const char *padding = "000000000";
int padLen;
memset(tempbufl, 0x0, 256);
strcat(tempbufl, strtok(rawinput.pf.hdr.dec_str, ":"));
strcat(tempbufl, strtok((char *) 0, ":"));


/* hack to fix unconventional dec_str in BL guppi raw gits */
memset(tempbuf, 0x0, 16);
strcat(tempbuf, strtok((char *) 0, ":"));
padLen = 7 - strlen(tempbuf); // Calc Padding length
if(padLen < 0) padLen = 0;    // Avoid negative length
sprintf(tempbufl + strlen(tempbufl), "%*.*s%s", padLen, padLen, padding, tempbuf);  // LEFT Padding 

src_dej = strtod(tempbufl, (char **) NULL);



if(vflag>=1) fprintf(stderr,"Writing filterbank headers...\n");
for(i=0;i<gpu_spec[0].nspec;i++){
	sprintf(gpu_spec[i].filename, "%s%04ld.fil",partfilename,i);
	if(!(gpu_spec[i].filterbank_file = fopen(gpu_spec[i].filename, "wb"))) {
	    perror(gpu_spec[i].filename);
	    exit(1);
	}

	foff =  fabs(rawinput.pf.hdr.df)/gpu_spec[i].cufftN * -1;
	nchans = gpu_spec[i].cufftN * rawinput.pf.hdr.nchan;
	tsamp = gpu_spec[i].cufftN/(fabs(rawinput.pf.hdr.df) * 1000000) * gpu_spec[i].integrationtime;
	fch1= rawinput.pf.hdr.fctr + fabs(rawinput.pf.hdr.BW)/2 + (0.5*foff) + hack;

	/* dump filterbank header */
	filterbank_header(gpu_spec[i].filterbank_file);
 }

 sprintf(filname, "%s.headers",partfilename);
 if(!(rawinput.headerfile = fopen(filname, "wb"))) {
     perror(filname);
     exit(1);
 }






do{
										
	if(!rawinput.invalid){						  
		  if(rawinput.fil == NULL) {
//		  if(filehandle < 0) {

			  /* no file is open for this band, try to open one */
			  sprintf(filname, "%s.%04d.raw",rawinput.file_prefix,rawinput.curfile);
			  printf("filename is %s\n",filname);
			  if(exists(filname)){
				 printf("opening %s\n",filname);				
				 if(!(rawinput.fil = fopen(filname, "rb"))) {
				     perror(filname);
				     exit(1);
				 }
				 //filehandle = open(filname, O_RDONLY);
				 if(rawinput.curfile == 0 && rawinput.first_file_skip != 0) fseek(rawinput.fil, rawinput.first_file_skip, SEEK_CUR);  
//				 if(rawinput.curfile == 0 && rawinput.first_file_skip != 0) lseek(filehandle, rawinput.first_file_skip, SEEK_CUR);  
			  }	else {
			  	rawinput.invalid = 1;
		  	  	printf("couldn't open any more files!\n");
		  	  }
		  }

	if(rawinput.fil){
//	if(filehandle > 0){

		if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768) {
//		if(read(filehandle, buf, 32768) == 32768) {
				
			fseek(rawinput.fil, -32768, SEEK_CUR);
//			lseek(filehandle, -32768, SEEK_CUR);

			if(vflag>1) fprintf(stderr, "header length: %d\n", gethlength(buf));
			
			guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);

			if(vflag>1) {
				 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
				 fprintf(stderr, "packetsize: %d\n", rawinput.gf.packetsize);
				 fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
				 fprintf(stderr, "n_dropped: %d\n",rawinput.gf.n_dropped);
				 fprintf(stderr, "RA: %f\n",rawinput.pf.sub.ra);
				 fprintf(stderr, "DEC: %f\n",rawinput.pf.sub.dec);
				 fprintf(stderr, "subintoffset %f\n", rawinput.pf.sub.offs);
				 fprintf(stderr, "tsubint %f\n", rawinput.pf.sub.tsubint);
			}
					
				  if(rawinput.gf.packetindex == curindx) {


					  /* read a subint with correct index, read the data */
					  if(rawinput.pf.hdr.directio == 0){
						  hlength = (long int) gethlength(buf);

						  /* write out header for archiving */
						  fwrite(buf, sizeof(char), hlength, rawinput.headerfile);

						  fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						  rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
 
								//lseek(filehandle, gethlength(buf), SEEK_CUR);				
 								//rv = read(filehandle, rawinput.pf.sub.data, rawinput.pf.sub.bytes_per_subint);
					  } else {
					  	  hlength = (long int) gethlength(buf);
					  	  
					  	  /* write out header for archiving */
						  fwrite(buf, sizeof(char), hlength, rawinput.headerfile);

					  	  if(vflag>1) fprintf(stderr, "header length: %ld\n", hlength);
						  if(vflag>1) fprintf(stderr, "seeking: %ld\n", hlength + ((512 - (hlength%512))%512) );
					  	  fseek(rawinput.fil, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);
							//lseek(filehandle, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);				

						    rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);

 							//rv = read(filehandle, rawinput.pf.sub.data, rawinput.pf.sub.bytes_per_subint);


						  fseek(rawinput.fil, ( (512 - (rawinput.pf.sub.bytes_per_subint%512))%512), SEEK_CUR);
						 //lseek(filehandle, ( (512 - (rawinput.pf.sub.bytes_per_subint%512))%512), SEEK_CUR);				

					  }
					  
					  if((long int)rv == rawinput.pf.sub.bytes_per_subint){
						 if(vflag>1) fprintf(stderr,"read %d bytes from %ld in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
						  
						 /* need to have each channel be contiguous */
						 /* copy in to buffer by an amount offset by the total channel offset + the offset within that channel */
						 /* need to make sure we only grab the non-overlapping piece */
						 for(i = 0; i < rawinput.pf.hdr.nchan; i++) {
							 memcpy(channelbuffer + (i * chanbytes) + (subint_cnt * chanbytes_subint), rawinput.pf.sub.data + (i * chanbytes_subint_total), chanbytes_subint);												
						 }
						 //memcpy(channelbuffer + (subint_cnt * rawinput.pf.sub.bytes_per_subint), rawinput.pf.sub.data, rawinput.pf.sub.bytes_per_subint);
						 subint_cnt++;
			  
			  
						 if(vflag>=1) fprintf(stderr, "copied %lld bytes subint cnt %ld\n", chanbytes * rawinput.pf.hdr.nchan, subint_cnt);
			  
			
					   } else {
						   rawinput.fil = NULL;
						   rawinput.invalid = 1;
						   fprintf(stderr,"ERR: couldn't read as much as the header said we could... assuming corruption and exiting...\n");
						   exit(1);
					   }
				   
			  
				   } else if(rawinput.gf.packetindex > curindx) {
						fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						/* read a subint with too high an indx, must have dropped a whole subintegration*/
						/* don't read the subint, but increment the subint counter and allow old data to be rechannelized */
						/* so that we maintain continuity in time... */
						subint_cnt++;
						//curindx = curindx + indxstep;
					   /* We'll get the current valid subintegration again during the next time through this loop */

			  
				   } else if(rawinput.gf.packetindex < curindx) {
						fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						/* somehow we were expecting a higher packet index than we got !?!? */

						/* we'll read past this subint and try again next time through */

						 if(rawinput.pf.hdr.directio == 0){
							 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
							 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
 
								   //lseek(filehandle, gethlength(buf), SEEK_CUR);				
								   //rv = read(filehandle, rawinput.pf.sub.data, rawinput.pf.sub.bytes_per_subint);
						 } else {
							 hlength = (long int) gethlength(buf);
							 if(vflag>1) fprintf(stderr, "header length: %ld\n", hlength);
							 if(vflag>1) fprintf(stderr, "seeking: %ld\n", hlength + ((512 - (hlength%512))%512) );
							 fseek(rawinput.fil, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);
							  //lseek(filehandle, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);				

							 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);

							   //rv = read(filehandle, rawinput.pf.sub.data, rawinput.pf.sub.bytes_per_subint);

							 fseek(rawinput.fil, ( (512 - (rawinput.pf.sub.bytes_per_subint%512))%512), SEEK_CUR);
							  //lseek(filehandle, ( (512 - (rawinput.pf.sub.bytes_per_subint%512))%512), SEEK_CUR);				

						 }
						 
						 curindx = curindx - indxstep;

				   }


				   if(subint_cnt == num_bufs) {
						if(vflag>1) fprintf(stderr, "calling gpu_channelize...");						 	
						subint_cnt=0;
						gpu_channelize(gpu_spec, nchannels, nsamples);
						if(vflag>1) fprintf(stderr, "waiting for accumulate...");						 	

						  pthread_join(accumwrite_th0, NULL);
						  pthread_join(accumwrite_th1, NULL);
						  pthread_join(accumwrite_th2, NULL);
						  pthread_join(accumwrite_th3, NULL);

	 //								 pthread_join(accumwrite_th0, NULL);
  
		   /* now copy spectra to spectrum */
		 
						 for (i=0;i<gpu_spec[0].nspec;i++)	{
							 if ((gpu_spec[i].spectraperchannel * gpu_spec[i].spectracnt) >= gpu_spec[i].integrationtime) {
								 memcpy(gpu_spec[i].spectrum, gpu_spec[i].spectra, nchannels * gpu_spec[i].spectraperchannel * gpu_spec[i].cufftN * sizeof(float));																						  
								 gpu_spec[i].triggerwrite = gpu_spec[i].spectracnt;
								 gpu_spec[i].spectracnt = 0;

								 /* launch accumulate / disk write thread */								 

								if (i == 3) {
								   pthread_create (&accumwrite_th3, NULL, (void * (*)(void*)) &accumulate_write_single, (void *) &gpu_spec[3]);
								} else if (i == 2) {
								   pthread_create (&accumwrite_th2, NULL, (void * (*)(void*)) &accumulate_write_single, (void *) &gpu_spec[2]);
								} else if (i == 1) {
								   pthread_create (&accumwrite_th1, NULL, (void * (*)(void*)) &accumulate_write_single, (void *) &gpu_spec[1]);
								} else if (i == 0) {
								   pthread_create (&accumwrite_th0, NULL, (void * (*)(void*)) &accumulate_write_single, (void *) &gpu_spec[0]);
								}

							 }
						 }

	 //								 pthread_create (&accumwrite_th0, NULL, (void *) &accumulate_write, (void *) &gpu_spec);												  									 				 
				   }


			   } else {

			   /* file open but couldn't read 32KB */
				  fclose(rawinput.fil);
				  rawinput.fil = NULL;
				  //close(filehandle);
				  //filehandle=-1;
				  rawinput.curfile++;						
			   }
		}			 	 	 
	}

										
	if(rawinput.fil != NULL) curindx = curindx + indxstep;
//	if(filehandle > 0) curindx = curindx + indxstep;


} while(!(rawinput.invalid));
	
	
	
	fprintf(stderr, "finishing up...\n");

	pthread_join(accumwrite_th0, NULL); 
	pthread_join(accumwrite_th1, NULL); 
	pthread_join(accumwrite_th2, NULL); 
	pthread_join(accumwrite_th3, NULL); 

	if(vflag>=1) fprintf(stderr, "bytes: %ld\n",by);
	

	
	if (rawinput.pf.sub.data) {
		 free(rawinput.pf.sub.data);
		 fprintf(stderr, "freed subint data buffer\n");
	}

	
//obs_length = (double) floor(channelbuffer_pos/fftlen) * fftlen * rawinput.pf.hdr.dt; // obs length in seconds
//printf("obs length: %g\n", obs_length);


printf("start time %15.15Lg end time %15.15Lg %s %s barycentric velocity %15.15g barycentric acceleration %15.15g \n", firstinput.pf.hdr.MJD_epoch, rawinput.pf.hdr.MJD_epoch, firstinput.pf.hdr.ra_str, firstinput.pf.hdr.dec_str, firstinput.baryv, firstinput.barya);

	
//	outputfile = fopen("out.bin", "w");
//	fwrite(gpu_spec[0].spectrum, sizeof(float), nchannels * gpu_spec[0].cufftN, outputfile);
//	fflush(outputfile);
//	fclose(outputfile);


	free(gpu_spec[0].channelbuffer);

	for(i=0;i<gpu_spec[0].nspec;i++){
		free(gpu_spec[i].spectrum);
		free(gpu_spec[i].spectra);	
		free(gpu_spec[i].bandpass);
		fprintf(stderr, "freed buffers for i = %ld...\n", i);
		fclose(gpu_spec[i].filterbank_file);
	}	
	
	fclose(rawinput.headerfile); 

    exit(0);


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




void accumulate_write(void *ptr){

struct gpu_spectrometer *gpu_spec;

gpu_spec = (struct gpu_spectrometer *) ptr;

unsigned long int i,j,k,l,m,n;
float * accumspectra;
unsigned long int nchannels;
unsigned long int pointsperchannel, pointsperintegration;
nchannels = gpu_spec[0].rawinput->pf.hdr.nchan;
unsigned long int adjustedintegrationtime;
unsigned long int chanoffset;
unsigned long int integrationpnts;
unsigned long int offseta;
unsigned long int fftlen;

	  for(i = 0; i < gpu_spec[0].nspec; i++){
	  
			if (gpu_spec[i].triggerwrite != 0){
										
			   if (gpu_spec[i].spectraperchannel > 1) {								 									 
				  //want to sum over all integration times...
				  adjustedintegrationtime = (gpu_spec[i].integrationtime/gpu_spec[i].triggerwrite);
				  pointsperchannel = gpu_spec[i].spectraperchannel * gpu_spec[i].cufftN;
				  pointsperintegration =  adjustedintegrationtime * gpu_spec[i].cufftN;	
                  fftlen = gpu_spec[i].cufftN;
				if(vflag>1) fprintf(stderr, "adj int time: %ld pnts per channel: %ld  pnts per integration: %ld\n", adjustedintegrationtime, pointsperchannel, pointsperintegration);
				if(vflag>1) fprintf(stderr, "spectraperchan %ld\n", gpu_spec[i].spectraperchannel);


				for (j = 0; j < nchannels; j = j + 1) {
				   chanoffset = j * pointsperchannel;
				   for (n = 0; n < gpu_spec[i].spectraperchannel; n = n + adjustedintegrationtime) {  								 	
					   integrationpnts = chanoffset + n * fftlen;							  
					   for (m = 1; m < adjustedintegrationtime; m = m + 1) {  								 	
						   offseta = integrationpnts + m * fftlen;							  	
						   for(k = 0; k < fftlen; k++) {	
								   gpu_spec[i].spectrum[(integrationpnts) + k] += gpu_spec[i].spectrum[(offseta) + k];						
						   }	
					   }
					}
				}


			   if(vflag>1)fprintf(stderr, "For instance %ld writing integrated spectra only\n", i);
				   accumspectra = (float *) malloc (fftlen * nchannels * sizeof(float));

				   for (n = 0; n < gpu_spec[i].spectraperchannel; n = n + adjustedintegrationtime) {  								 	
					   offseta = n * fftlen;
					   for (j = 0; j < nchannels; j = j + 1) {						
						memcpy(accumspectra + (j * fftlen) , gpu_spec[i].spectrum + ( offseta + (pointsperchannel * j) ), sizeof(float) * fftlen);
						//fwrite(gpu_spec[i].spectrum + (offseta + (pointsperchannel * j) ), sizeof(float), fftlen, gpu_spec[i].filterbank_file);		
						}
						fwrite(accumspectra, sizeof(float), fftlen * nchannels, gpu_spec[i].filterbank_file);
				  
				   }			
				   free(accumspectra);

				/* DUMP TO FILE */																 
				
			   } else {								 									 

					 if(vflag>1)fprintf(stderr, "For instance %ld writing complete spectra\n", i);

					fwrite(gpu_spec[i].spectrum, sizeof(float), gpu_spec[i].cufftN * nchannels, gpu_spec[i].filterbank_file);		

					  /* DUMP TO FILE */																 
			   }

				gpu_spec[i].triggerwrite = 0;

			}

	  }


}


void accumulate_write_single(void *ptr){

	 struct gpu_spectrometer *gpu_spec;

	 gpu_spec = (struct gpu_spectrometer *) ptr;


	 unsigned long int i,j,k,l,m,n;
	 float * accumspectra;
	 unsigned long int nchannels;
	 unsigned long int pointsperchannel, pointsperintegration;
	 nchannels = gpu_spec->rawinput->pf.hdr.nchan;
	 unsigned long int adjustedintegrationtime;
	 unsigned long int chanoffset;
	 unsigned long int integrationpnts;
	 unsigned long int offseta;
	 unsigned long int fftlen;

	  
	 if (gpu_spec->triggerwrite != 0){
						   
	   if (gpu_spec->spectraperchannel > 1) {								 									 
		  //want to sum over all integration times...
		  adjustedintegrationtime = (gpu_spec->integrationtime/gpu_spec->triggerwrite);
		  pointsperchannel = gpu_spec->spectraperchannel * gpu_spec->cufftN;
		  pointsperintegration =  adjustedintegrationtime * gpu_spec->cufftN;	
		  fftlen = gpu_spec->cufftN;
		if(vflag>1) fprintf(stderr, "adj int time: %ld pnts per channel: %ld  pnts per integration: %ld\n", adjustedintegrationtime, pointsperchannel, pointsperintegration);
		if(vflag>1) fprintf(stderr, "spectraperchan %ld\n", gpu_spec->spectraperchannel);


		 for (j = 0; j < nchannels; j = j + 1) {
			chanoffset = j * pointsperchannel;
			for (n = 0; n < gpu_spec->spectraperchannel; n = n + adjustedintegrationtime) {  								 	
				integrationpnts = chanoffset + n * fftlen;							  
				for (m = 1; m < adjustedintegrationtime; m = m + 1) {  								 	
					offseta = integrationpnts + m * fftlen;							  	
					for(k = 0; k < fftlen; k++) {	
							gpu_spec->spectrum[(integrationpnts) + k] += gpu_spec->spectrum[(offseta) + k];						

					}	
				}
			 }
		 }

		if(vflag>1)fprintf(stderr, "For instance %ld writing integrated spectra only\n", i);
		 accumspectra = (float *) malloc (fftlen * nchannels * sizeof(float));

		  for (n = 0; n < gpu_spec->spectraperchannel; n = n + adjustedintegrationtime) {  								 	
			  offseta = n * fftlen;
			  for (j = 0; j < nchannels; j = j + 1) {						
				   memcpy(accumspectra + (j * fftlen) , gpu_spec->spectrum + ( offseta + (pointsperchannel * j) ), sizeof(float) * fftlen);
			  }
		 
			  fwrite(accumspectra, sizeof(float), fftlen * nchannels, gpu_spec->filterbank_file);
	  
		  }			
		  free(accumspectra);
   
	   } else {								 									 

		  if(vflag>1)fprintf(stderr, "For instance %ld writing complete spectra\n", i);
		  fwrite(gpu_spec->spectrum, sizeof(float), gpu_spec->cufftN * nchannels, gpu_spec->filterbank_file);		

	   }

		gpu_spec->triggerwrite = 0;
   
	 }

}



void gpu_channelize(struct gpu_spectrometer gpu_spec[4], long int nchannels, long long int nsamples)
{

	 long int i,j,k;
	 long int nframes;

/* chan 0, pol 0, r, pol 0 i, pol 1 ... */

	  i=0;
	 if(vflag>1) fprintf(stderr, "center_freq: %f\n\n", gpu_spec[i].rawinput->pf.hdr.fctr);
	 nframes = nsamples / gpu_spec[i].cufftN;

	 if(vflag>1) fprintf(stderr, "%ld\n", (size_t) nsamples * nchannels);
	 cudaThreadSynchronize();

	 /* copy whole subint onto gpu */

	 /* explode to a floating point array twice the length of nsamples, one for each polarization */
	 if(gpu_spec[i].rawinput->pf.hdr.nbits == 2) {
	 	 HANDLE_ERROR( cudaMemcpy( gpu_spec[i].channelbufferd, gpu_spec[0].channelbuffer, (size_t) nsamples * nchannels, cudaMemcpyHostToDevice) ); 
		 explode_wrapper(gpu_spec[i].channelbufferd, gpu_spec[i].a_d, nsamples * nchannels);
	 } else if (gpu_spec[i].rawinput->pf.hdr.nbits == 8) {
	 	 //HANDLE_ERROR( cudaMemcpy( gpu_spec[i].channelbufferd8, gpu_spec[0].channelbuffer, (size_t) nsamples * nchannels * 4, cudaMemcpyHostToDevice) ); 
		 //explode8simple_wrapper(gpu_spec[i].channelbufferd8, gpu_spec[i].a_d, nsamples * nchannels); 
		 //explode8_wrapper(gpu_spec[i].channelbufferd8, gpu_spec[i].a_d, nsamples * nchannels); 

	 	 HANDLE_ERROR( cudaMemcpy( gpu_spec[i].channelbufferd, gpu_spec[0].channelbuffer, (size_t) nsamples * nchannels * 4, cudaMemcpyHostToDevice) ); 
		 explode8lut_wrapper(gpu_spec[i].channelbufferd, gpu_spec[i].a_d, nsamples * nchannels); 

	 }	

//	 cufftComplex tempcmplx[2048];
//	HANDLE_ERROR( cudaMemcpy(tempcmplx, gpu_spec[i].a_d, 2048 * sizeof(cufftComplex), cudaMemcpyDeviceToHost) );
//    for(i=128;i<256;i=i+4) fprintf(stderr, "%08x %08x\n", gpu_spec[0].channelbuffer[i], gpu_spec[0].channelbuffer[i+1]);
//    for(i=32;i<96;i++) fprintf(stderr, "%f %f\n", tempcmplx[i].x, tempcmplx[i].y);
//	 exit(1);

	 cudaThreadSynchronize();
	 for(i=0;i<gpu_spec[0].nspec;i++){

			if ((gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN) > MAXSIZE) {

				HANDLE_CUFFT_ERROR( cufftExecC2C(gpu_spec[i].plan, gpu_spec[i].a_d, gpu_spec[i].b_d, CUFFT_FORWARD) );
				HANDLE_CUFFT_ERROR( cufftExecC2C(gpu_spec[i].plan, gpu_spec[i].a_d + ((gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN) / 2), gpu_spec[i].b_d + ((gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN) / 2), CUFFT_FORWARD) );

			} else {

				HANDLE_CUFFT_ERROR( cufftExecC2C(gpu_spec[i].plan, gpu_spec[i].a_d, gpu_spec[i].b_d, CUFFT_FORWARD) );
			
			}

			if (gpu_spec[i].spectracnt == 0) HANDLE_ERROR( cudaMemset(gpu_spec[i].spectrumd, 0x0, gpu_spec[i].cufftbatchSize * gpu_spec[i].cufftN * sizeof(float)) );

			if(gpu_spec[i].pol == 2) {
				detect_wrapper(gpu_spec[i].b_d, nsamples * nchannels, gpu_spec[i].cufftN, gpu_spec[i].bandpassd, gpu_spec[i].spectrumd);
			} else if (gpu_spec[i].pol == 1) {
					detectY_wrapper(gpu_spec[i].b_d, nsamples * nchannels, gpu_spec[i].cufftN, gpu_spec[i].bandpassd, gpu_spec[i].spectrumd);
			} else if (gpu_spec[i].pol == 0) {
					detectX_wrapper(gpu_spec[i].b_d, nsamples * nchannels, gpu_spec[i].cufftN, gpu_spec[i].bandpassd, gpu_spec[i].spectrumd);
			}			
			

			gpu_spec[i].spectracnt++; 
	   
			cudaThreadSynchronize();
			if ((gpu_spec[i].spectraperchannel * gpu_spec[i].spectracnt) >= gpu_spec[i].integrationtime) {
				HANDLE_ERROR( cudaMemcpy(gpu_spec[i].spectra, gpu_spec[i].spectrumd, nsamples * nchannels * sizeof(float), cudaMemcpyDeviceToHost) );
			}
			if(vflag>1) fprintf(stderr, "\n\n%f %f\n\n", gpu_spec[i].spectra[100], gpu_spec[i].spectra[1]);
	}

/*
for (i=0;i<gpu_spec[0].nspec;i++)	{
	 if (gpu_spec[i].spectraperchannel >= gpu_spec[i].integrationtime) {								 									 

		  memcpy(gpu_spec[i].spectrum, gpu_spec[i].spectra, nchannels * gpu_spec[i].spectraperchannel * gpu_spec[i].cufftN * sizeof(float));																						  

	 } else {

		 for(j = 0; j < (nchannels * gpu_spec[i].spectraperchannel * gpu_spec[i].cufftN); j++) {
			  gpu_spec[i].spectrum[j] = gpu_spec[i].spectrum[j] + gpu_spec[i].spectra[j]; 						
		 }	
		 
	}
	if ((gpu_spec[i].spectraperchannel * gpu_spec[i].spectracnt) >= gpu_spec[i].integrationtime) {
		gpu_spec[i].triggerwrite = gpu_spec[i].spectracnt;
		gpu_spec[i].spectracnt = 0;
	}
}
*/	
	
	
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

//for(i=0;i<chanbytes * nchans;i++) {
//fprintf(outputfile, "%f\n", gpu_spec->spectra[i]);
//}

/*
sprintf(tempfilname, "%s/%s.%f.fits",gpu_spec->scratchpath, gpu_spec->rawinput->pf.hdr.source, gpu_spec->rawinput->pf.hdr.fctr);
outputfile = fopen(tempfilname, "a+");

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
*/
//fwrite(gpu_spec->spectra, sizeof(char), chanbytes * nchans * sizeof(float), outputfile);



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











void filterbank_header(FILE *outptr) /* includefile */
{
  if (sigproc_verbose)
    fprintf (stderr, "sigproc::filterbank_header\n");

  int i,j;
  output=outptr;

  if (obits == -1) obits=nbits;
  /* go no further here if not interested in header parameters */

  if (headerless)
  {
    if (sigproc_verbose)
      fprintf (stderr, "sigproc::filterbank_header headerless - abort\n");
    return;
  }

  
  if (sigproc_verbose)
    fprintf (stderr, "sigproc::filterbank_header HEADER_START");

  send_string("HEADER_START");
  send_string("rawdatafile");
  send_string(inpfile);
  if (!strings_equal(source_name,""))
  {
    send_string("source_name");
    send_string(source_name);
  }
    send_int("machine_id",machine_id);
    send_int("telescope_id",telescope_id);
    send_coords(src_raj,src_dej,az_start,za_start);
    if (zerolagdump) {
      /* time series data DM=0.0 */
      send_int("data_type",2);
      refdm=0.0;
      send_double("refdm",refdm);
      send_int("nchans",1);
    } else {
      /* filterbank data */
      send_int("data_type",1);
      send_double("fch1",fch1);
      send_double("foff",foff);
      send_int("nchans",nchans);
    }
    /* beam info */
    send_int("nbeams",nbeams);
    send_int("ibeam",ibeam);
    /* number of bits per sample */
    send_int("nbits",obits);
    /* start time and sample interval */
    send_double("tstart",tstart+(double)start_time/86400.0);
    send_double("tsamp",tsamp);
    if (sumifs) {
      send_int("nifs",1);
    } else {
      j=0;
      for (i=1;i<=nifs;i++) if (ifstream[i-1]=='Y') j++;
      if (j==0) error_message("no valid IF streams selected!");
      send_int("nifs",j);
    }
    send_string("HEADER_END");
}

void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  if (swapout) swap_int(&len);
  fwrite(&len, sizeof(int), 1, output);
  if (swapout) swap_int(&len);
  fwrite(string, sizeof(char), len, output);
  /*fprintf(stderr,"%s\n",string);*/
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  if (swapout) swap_float(&floating_point);
  fwrite(&floating_point,sizeof(float),1,output);
  /*fprintf(stderr,"%f\n",floating_point);*/
}

void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  if (swapout) swap_double(&double_precision);
  fwrite(&double_precision,sizeof(double),1,output);
  /*fprintf(stderr,"%f\n",double_precision);*/
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  if (swapout) swap_int(&integer);
  fwrite(&integer,sizeof(int),1,output);
  /*fprintf(stderr,"%d\n",integer);*/
}

void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  if (swapout) swap_long(&integer);
  fwrite(&integer,sizeof(long),1,output);
  /*fprintf(stderr,"%ld\n",integer);*/
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}

  
/* 
	some useful routines written by Jeff Hagen for swapping
	bytes of data between Big Endian and  Little Endian formats:
*/
void swap_short( unsigned short *ps ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = ( unsigned char *)ps;
  t = pc[0];
  pc[0] = pc[1];
  pc[1] = t;
}

void swap_int( int *pi ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pi;

  t = pc[0];
  pc[0] = pc[3];
  pc[3] = t;

  t = pc[1];
  pc[1] = pc[2];
  pc[2] = t;
}

void swap_float( float *pf ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pf;

  t = pc[0];
  pc[0] = pc[3];
  pc[3] = t;

  t = pc[1];
  pc[1] = pc[2];
  pc[2] = t;
}

void swap_ulong( unsigned long *pi ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pi;

  t = pc[0];
  pc[0] = pc[3];
  pc[3] = t;

  t = pc[1];
  pc[1] = pc[2];
  pc[2] = t;
}

void swap_long( long *pi ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pi;

  t = pc[0];
  pc[0] = pc[3];
  pc[3] = t;

  t = pc[1];
  pc[1] = pc[2];
  pc[2] = t;
}

void swap_double( double *pd ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pd;

  t = pc[0];
  pc[0] = pc[7];
  pc[7] = t;

  t = pc[1];
  pc[1] = pc[6];
  pc[6] = t;

  t = pc[2];
  pc[2] = pc[5];
  pc[5] = t;

  t = pc[3];
  pc[3] = pc[4];
  pc[4] = t;

}

void swap_longlong( long long *pl ) /* includefile */
{
  unsigned char t;
  unsigned char *pc;

  pc = (unsigned char *)pl;

  t = pc[0];
  pc[0] = pc[7];
  pc[7] = t;

  t = pc[1];
  pc[1] = pc[6];
  pc[6] = t;

  t = pc[2];
  pc[2] = pc[5];
  pc[5] = t;

  t = pc[3];
  pc[3] = pc[4];
  pc[4] = t;
}

int little_endian() /*includefile*/
{
  char *ostype;

  if((ostype = (char *)getenv("OSTYPE")) == NULL )
    error_message("environment variable OSTYPE not set!");
  if (strings_equal(ostype,"linux")) return 1;
  if (strings_equal(ostype,"hpux")) return 0;
  if (strings_equal(ostype,"solaris")) return 0;
  if (strings_equal(ostype,"darwin")) return 0;
  fprintf(stderr,"Your OSTYPE environment variable is defined but not recognized!\n");
  fprintf(stderr,"Consult and edit little_endian in swap_bytes.c and then recompile\n");
  fprintf(stderr,"the code if necessary... \n");
  exit(0);
}



