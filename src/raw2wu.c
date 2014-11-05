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
#include "barycenter.h"
#include "rawdopplersearch.h"

/* Guppi channel-frequency mapping */
/* sample at 1600 MSamp for an 800 MHz BW */
/* N = 256 channels with frequency centers at m * fs/N */
/* m = 0,1,2,3,... 255 */

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);

/* prototypes */

int exists(const char *fname);

int unpack_samples(unsigned char * raw, unsigned char *dest, long int count, int pol);

long int read_guppi_blocks(unsigned char **setibuffer, long int startblock, long int numblocks, int channel, int polarization, int vflag, char * file_prefix);


int main(int argc, char *argv[]) {


	/* header structure */
	
	struct seti_data {
        int               	header_size;    //                                                            need for gbt data
        int               	data_size;      //                                                            need for gbt data
        char              	name[36];       //                                                            need for gbt data
		int               	channel;        //                                                            need for gbt data
		int					polarization;    // which pol?  0 or 1
        double            	data_time;      // time stamp : the final data sample in this block           need for gbt data
        double            	coord_time;     // time stamp : ra/dec (ie, the time of Az/Za acquisition)    need for gbt data - 
        double            	ra;             // ra for the beam from which the data in this block came     need for gbt data
        double            	dec;            // dec for the beam from which the data in this block came    need for gbt data
        long              	sky_freq;       // aka center frequency, Hz                                   need for gbt data
        double            	samplerate;     //                                                            need for gbt data		
	};

	struct seti_data setiheader;

	float samples[10];		
	int i=0;


	/* function input variables */
	unsigned char *setibuffer=NULL;
	long int startblock=10;
	long int numblocks=5;
	int channel = 3;
	int polarization = 0;
	int vflag=2; //verbose
	char file_prefix[1024];
	
	long int blocksread;
	sprintf(file_prefix, "/disks/sting/kepler_disk_10/disk_09/gpu6/gpu6_guppi_55691_KID8891318_A_0040.0000.raw");	
	
	blocksread = read_guppi_blocks(&setibuffer, startblock, numblocks, channel, polarization, vflag, file_prefix);
    
	printf("Read %ld blocks!\n", blocksread);
	
	memcpy(&setiheader, setibuffer, sizeof(setiheader));

	printf("Header Size: %d\n", setiheader.header_size);
	printf("Data Size: %d\n", setiheader.data_size);		
	printf("Data Time: %f\n", setiheader.data_time);
	printf("Coordinate Time: %f\n", setiheader.coord_time);
	printf("RA: %f\n", setiheader.ra);
	printf("Dec: %f\n", setiheader.dec);

	memcpy(samples, setibuffer+setiheader.header_size, sizeof(float) * 10);
	for(i=0;i<10;i=i+2) printf("Samples: %f + %f i\n", samples[i], samples[i+1] );

	printf("\nBlock 2:\n");
	memcpy(&setiheader, setibuffer + setiheader.header_size + setiheader.data_size, sizeof(setiheader));
	printf("Header Size: %d\n", setiheader.header_size);
	printf("Data Size: %d\n", setiheader.data_size);		
	printf("Data Time: %f\n", setiheader.data_time);
	printf("Coordinate Time: %f\n", setiheader.coord_time);
	printf("RA: %f\n", setiheader.ra);
	printf("Dec: %f\n", setiheader.dec);
    

exit(0);
}


long int read_guppi_blocks(unsigned char **setibuffer, long int startblock, long int numblocks, int channel, int polarization, int vflag, char * file_prefix) {


	int overlap; // amount of overlap between sub integrations in samples

	int filecnt=0;
    char buf[32768];
	

	unsigned char *channelbuffer;	

	char filname[250];

	struct gpu_input rawinput;	
	
	struct seti_data {
        int               header_size;    //                                                            need for gbt data
        int               data_size;      //                                                            need for gbt data
        char              name[36];       //                                                            need for gbt data
        int               channel;        //                                                            need for gbt data
		int 			  polarization;    // which pol?  0 or 1
        double            data_time;      // time stamp : the final data sample in this block           need for gbt data
        double            coord_time;     // time stamp : ra/dec (ie, the time of Az/Za acquisition)    need for gbt data - 
        double            ra;             // ra for the beam from which the data in this block came     need for gbt data
        double            dec;            // dec for the beam from which the data in this block came    need for gbt data
        long              sky_freq;       // aka center frequency, Hz                                   need for gbt data
        double            samplerate;     //                                                            need for gbt data
		
	};

	
	struct seti_data setiheader;
	
	
	
	long long int startindx;
	long long int curindx;
	long long int chanbytes=0;
	long long int chanbytes_overlap = 0;
	long long int subint_offset = 0;

	long long int setibuffer_pos = 0;



	long int currentblock=0;
	
	
	int indxstep = 0;

	    
	size_t rv=0;
	long unsigned int by=0;
    

	long int i=0,j=0;

    
    


    
	rawinput.file_prefix = NULL;
	rawinput.fil = NULL;
	rawinput.invalid = 0;
	rawinput.first_file_skip = 0;  


       opterr = 0;
rawinput.file_prefix=file_prefix;
     

/* no input specified */
if(rawinput.file_prefix == NULL) {
	printf("WARNING no input stem specified%ld\n", (i+1));
	return 0;
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
		   
		   if(rawinput.pf.hdr.nbits == 8) {
			  
			  fprintf(stderr, "caught an 8 bit header\n");
			  
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
	  		fprintf(stderr, "couldn't read a header\n");
			return 0;
	  }
} else {
	fprintf(stderr, "couldn't open first file\n");
	return 0;
}


if(vflag>=1) fprintf(stderr, "calculating index step\n");

/* number of packets that we *should* increment by */
indxstep = (int) ((rawinput.pf.sub.bytes_per_subint * 4) / rawinput.gf.packetsize) - (int) (rawinput.overlap * rawinput.pf.hdr.nchan * rawinput.pf.hdr.rcvr_polns * 2 / rawinput.gf.packetsize);


//spectraperint = indxstep * band[first_good_band].gf.packetsize / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 * ftacc);
//spectraperint = ((rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan) - rawinput.overlap) / ftacc;	

overlap = rawinput.overlap;

if (channel >= rawinput.pf.hdr.nchan) {
	fprintf(stderr, "channel %d more than channels in data %d\n", channel, rawinput.pf.hdr.nchan);
	return 0;
} else {
	fprintf(stderr, "Numer of channels in file %d\n", rawinput.pf.hdr.nchan);
}

/* number of non-overlapping bytes in each channel */
/* indxstep increments by the number of unique packets in each sub-integration */
/* packetsize is computed based on the original 8 bit resolution */
/* divide by 4 to get to 2 bits, nchan to get to number of channels */

chanbytes = indxstep * rawinput.gf.packetsize / (4 * rawinput.pf.hdr.nchan); 
fprintf(stderr, "chan bytes %lld\n", chanbytes);

channelbuffer  = (unsigned char *) calloc( chanbytes , sizeof(char) );
if (!channelbuffer) {
	fprintf(stderr, "error: couldn't allocate memory for buffer\n");
	return 0;
}

fprintf(stderr, "malloc'ed %Ld bytes for channel %d\n",  chanbytes, channel );	




//	tstart=band[first_good_band].pf.hdr.MJD_epoch;
//	tsamp = band[first_good_band].pf.hdr.dt * ftacc;

//	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
//	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));

	

/* size of this header */
setiheader.header_size = sizeof(setiheader);

/* number of samples * size of a float * 2 for real/imag */
setiheader.data_size = (chanbytes * sizeof(float) * 2);

/* populate with source name from psrfits header */
sprintf(setiheader.name, "%s", rawinput.pf.hdr.source);

/* populate channel and polarization */
setiheader.channel = channel;
setiheader.polarization = polarization;

/* (complex) sample rate in Hz */
setiheader.samplerate = rawinput.pf.hdr.df * 1000000;

/* sky frequency in Hz */
setiheader.sky_freq = (long int) ((double) 1000000 * ((rawinput.pf.hdr.fctr - (rawinput.pf.hdr.BW/2)) + ((channel+0.5) * rawinput.pf.hdr.df)));

/*
printf("freq %ld\n", setiheader.sky_freq);
printf("fctr %lf\n", rawinput.pf.hdr.fctr);
printf("bandwidth %lf\n", rawinput.pf.hdr.BW);
printf("freqdouble %lf\n", ((double) 1000000 * ((rawinput.pf.hdr.fctr - (rawinput.pf.hdr.BW/2)) + ((channel+0.5) * rawinput.pf.hdr.df))));
*/



*setibuffer = (unsigned char *) calloc( ((chanbytes * sizeof(float) * 2) + setiheader.header_size) * numblocks, sizeof(char) ); 

if (!*setibuffer) {
	fprintf(stderr, "error: couldn't allocate memory for buffer\n");
	return 0;
}

if(vflag>=1) fprintf(stderr, "malloced %Ld bytes for storing channel %d\n", ((chanbytes * sizeof(float) * 2) + setiheader.header_size) * numblocks, channel );	


/* total number of bytes per channel, including overlap */
chanbytes_overlap = rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan;


/* memory offset for our chosen channel within a subint */
subint_offset = channel * chanbytes_overlap;


if(vflag>=1) fprintf(stderr, "Index step: %d\n", indxstep);
if(vflag>=1) fprintf(stderr, "bytes per subint %d\n",rawinput.pf.sub.bytes_per_subint );



fflush(stdout);


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
		  	  }
		  }

		  if(rawinput.fil){		  
		  
				if((fread(buf, sizeof(char), 32768, rawinput.fil) == 32768) && (currentblock < (numblocks + startblock - 1)) ) {				
					fseek(rawinput.fil, -32768, SEEK_CUR);
					if(vflag>=1) fprintf(stderr, "header length: %d\n", gethlength(buf));
					guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);

					currentblock = (long int) ((double) rawinput.gf.packetindex/ (double) indxstep);

					if(vflag>=1) {
						 fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
						 fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
						 fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
						 fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
						 fprintf(stderr, "blocks: %ld\n", currentblock);
						 fprintf(stderr, "RA: %f\n\n",rawinput.pf.sub.ra);
						 fprintf(stderr, "DEC: %f\n\n",rawinput.pf.sub.dec);
						 fprintf(stderr, "subintoffset %f\n", rawinput.pf.sub.offs);
						 fprintf(stderr, "tsubint %f\n", rawinput.pf.sub.tsubint);
						 fprintf(stderr, "MJD %Lf\n", rawinput.pf.hdr.MJD_epoch);
					}
					
					/* populate the variables that change with each block */
				    /* RA (J2000) at subint centre (deg), Dec (J2000) at subint centre (deg), time in MJD */
				    
					/* increment time for data by 0.5 x length of block to push quoted time to the _last_ sample in the block */					
					setiheader.data_time = rawinput.pf.hdr.MJD_epoch + ((rawinput.pf.sub.offs + rawinput.pf.sub.tsubint/2)/86400.0) ;     
					setiheader.coord_time = rawinput.pf.hdr.MJD_epoch + (rawinput.pf.sub.offs/86400.0);    
										
					setiheader.ra = rawinput.pf.sub.ra;            
					setiheader.dec = rawinput.pf.sub.dec;           

					
					
			   		if(rawinput.gf.packetindex == curindx) {
						 /* read a subint with correct index */

						 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						 rv=0;
						 
						 if (currentblock > (startblock-1)){
						 
						 	rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
						 	
							if( ((long int)rv == rawinput.pf.sub.bytes_per_subint) ){
							   if(vflag>=1) fprintf(stderr,"read %d bytes from %ld in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);						   
	   
		
							   memcpy((*setibuffer)+setibuffer_pos, &setiheader, sizeof(setiheader));
							   setibuffer_pos = setibuffer_pos + sizeof(setiheader);
							   unpack_samples(rawinput.pf.sub.data + subint_offset, (*setibuffer) + setibuffer_pos, chanbytes, polarization);					   
							   setibuffer_pos = setibuffer_pos + (chanbytes * 2 * sizeof(float));
							   
							   	if(vflag>=1) fprintf(stderr,"buffer position: %Ld\n", setibuffer_pos);						   

							   //memcpy(channelbuffer + channelbuffer_pos, rawinput.pf.sub.data + subint_offset, chanbytes);										   

							   //channelbuffer_pos = channelbuffer_pos + chanbytes;

							} else {
								rawinput.fil = NULL;
								rawinput.invalid = 1;
								fprintf(stderr,"ERR: couldn't read as much as the header said we could... assuming corruption and exiting...\n");
								return 0;
							}

						 } else (rv = fseek(rawinput.fil, rawinput.pf.sub.bytes_per_subint, SEEK_CUR)); 
						 
					
					} else if( (rawinput.gf.packetindex > curindx) && (currentblock > (startblock-1)) ) {

						 fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						 /* read a subint with too high an indx, must have dropped a whole subintegration*/
				

						/* pf.sub.data *should* still contain the last valid subint */
						/* grab a copy of the last subint  - probably should add gaussian noise here, but this is better than nothing! */

						/* we'll keep the ra/dec values from this subint, but push back the time to keep everything sensible */
					    setiheader.data_time = rawinput.pf.hdr.MJD_epoch + ((rawinput.pf.sub.offs - rawinput.pf.sub.tsubint/2)/86400.0) ;     

						memcpy((*setibuffer) + setibuffer_pos, &setiheader, sizeof(setiheader));
						setibuffer_pos = setibuffer_pos + sizeof(setiheader);
						memmove((*setibuffer) + setibuffer_pos, (*setibuffer) + setibuffer_pos - sizeof(setiheader) - chanbytes, (chanbytes * 2 * sizeof(float)));					   
						setibuffer_pos = setibuffer_pos + (chanbytes * 2 * sizeof(float));

						/* We'll get the current valid subintegration again on the next time through this loop */


					} else if(rawinput.gf.packetindex < curindx) {
						 fprintf(stderr,"Error expecting a higher packet index than we got curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );
						 /* somehow we were expecting a higher packet index than we got !?!? */	
						 fprintf(stderr, "assuming corruption and exiting...\n");
						 return(0);
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
	
if (rawinput.pf.sub.data) {
	 free(rawinput.pf.sub.data);
	 fprintf(stderr, "freed subint data buffer\n");
}


fprintf(stderr, "done\n");

free(channelbuffer);

if(vflag>=1) fprintf(stderr, "grabbed %Ld blocks \n",setibuffer_pos / ((chanbytes * 2 * sizeof(float)) + sizeof(setiheader)));
return setibuffer_pos / ((chanbytes * 2 * sizeof(float)) + sizeof(setiheader));


fprintf(stderr, "closed output file...\n");

}





int unpack_samples(unsigned char * raw, unsigned char * dest, long int count, int pol)
{
/* unpack guppi samples stored in raw into dest as 32 bit floats real/imag */
/* count is number of samples, offset is a memory offset within dest (bytes), pol is polarization (0 for X, 1 for Y) */

	 int i;

	 float * samples;
	 float quantlookup[4];

	 samples = malloc(sizeof(float) * count * 2);

	 quantlookup[0] = 3.3358750;
	 quantlookup[1] = 1.0;
	 quantlookup[2] = -1.0;
	 quantlookup[3] = -3.3358750;

	 for(i=0;i<count;i=i+2) {

		  /* real */
		  samples[i] = quantlookup[( raw[i] >> (0 * 2) & 1) +  (2 * (raw[i] >> (0 * 2 + 1) & 1))]   ; //real pol 0

		  /* imag */
		  samples[i+1] = quantlookup[(raw[i] >> (1 * 2) & 1) +  (2 * (raw[i] >> (1 * 2 + 1) & 1))]   ; //imag pol 0

	 }
     fprintf(stderr, "in func\n");
	 memcpy(dest, samples, sizeof(float) * count * 2);
     fprintf(stderr, "in func\n");

     free(samples);
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





