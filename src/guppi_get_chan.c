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
#include "fcntl.h"


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
};



		
long int simple_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);
		
long int extension_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);

/* prototypes */

float log2f(float x);
double log2(double x);
long int lrint(double x);




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
	 int nBytes;
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
     
       while ((c = getopt (argc, argv, "Vvi:o:h:b:B:")) != -1)
         switch (c)
           {
           case 'v':
             vflag = 1;
             break;
           case 'B':
             num_bufs = atoi(optarg);
             break;                          
           case 'b':
             inbits = atoi(optarg);
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



if(strstr(rawinput.file_prefix, ".0000.raw") != NULL) memset(rawinput.file_prefix + strlen(rawinput.file_prefix) - 9, 0x0, 9);




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


nchannels = rawinput.pf.hdr.nchan;

overlap = rawinput.overlap;

/* number of non-overlapping bytes in each channel */
/* indxstep increments by the number of unique packets in each sub-integration */
/* packetsize is computed based on the original 8 bit resolution */
/* divide by nchan to get to bytes/channel */
/* divide by 4 to get back to 2 bits */


chanbytes = (indxstep * rawinput.gf.packetsize / ((8/rawinput.pf.hdr.nbits) * rawinput.pf.hdr.nchan)) * num_bufs; 
if(vflag>1) fprintf(stderr, "number of non-overlapping bytes for each chan %lld\n", chanbytes);

/* for 2 bit data, the number of dual pol samples is the same as the number of bytes */

nsamples = chanbytes / 4 * (8/rawinput.pf.hdr.nbits);



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




/* load lookup table into GPU memory */

if (rawinput.pf.hdr.nbits == 2) {

	float lookup[4];
	 lookup[0] = 3.3358750;
	 lookup[1] = 1.0;
	 lookup[2] = -1.0;
	 lookup[3] = -3.3358750;

} else if (rawinput.pf.hdr.nbits == 8) {
	if(vflag>0) fprintf(stderr,"Initializing for 8 bits... chanbytes: %ld nchannels: %ld\n", chanbytes, nchannels);

	/* try unsigned for lookup table... */
	float lookup[256];
	for(i=0;i<128;i++) lookup[i] = (float) i;
	for(i=128;i<256;i++) lookup[i] = (float) (i - 256);
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
				 rawinput.fil = fopen(filname, "rb");			 
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

			HANDLE_ERROR( cufftExecC2C(gpu_spec[i].plan, gpu_spec[i].a_d, gpu_spec[i].b_d, CUFFT_FORWARD) ); 

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



