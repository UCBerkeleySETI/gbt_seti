#include <stdio.h>
#include <stdlib.h>
#include "psrfits.h"
#include "guppi_params.h"
#include "fitshead.h"
#include <math.h>
#include <arpa/inet.h>
#include <string.h>

/* raw_ascii.c */
/* output 2 or 8 bit complex raw voltage data from GUPPI as ASCII txt in the following 		*/
/* format, where each value below is labeled by: 											*/
/* time:channel:polarization:(r=real, i=imaginary)											*/
/* e.g. "0:4:0:r" indicates time 0, channel 4, polarization 0, real part					*/
/* or,	"0:4:0:i" indicates time 0, channel 4, polarization 0, imaginary part				*/
/*																							*/
/* for times 0 to M, channels 0 to N, polarizations 0 and 1 								*/
/* 0:0:0:r, 0:0:0:i, 0:0:1:r, 0:0:1:i, 0:1:0:r, 0:1:0:i, 0:1:1:r, 0:1:1:i, ... , 0:N:1:i  	*/
/* ...																						*/
/* M:0:0:r, M:0:0:i, M:0:1:r, M:0:1:i, M:1:0:r, M:1:0:i, M:1:1:r, M:1:1:i, ... , M:N:1:i  	*/
/*																							*/
/*																							*/
/* in other words, each line gives all samples for each time in channel order. 				*/
/*																							*/




/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);
double round(double x);
                       
          
              
int main(int argc, char *argv[]) {
	struct guppi_params gf;
    struct psrfits pf;
    char buf[32768];
    char partfilename[250]; //file name for first part of file
    char quantfilename[250]; //file name for first part of file
    char keywrd[250];
	int subintcnt=0;
	int filepos=0;
	size_t rv=0;
	int by=0;
    
    FILE *fil;   //input file
    FILE *partfil;  //partial file
    FILE *quantfil;  //quantized file
    
	int a,x,y,z;
	double power;
	int sampsper = 8192;
	int sample;
	
	double running_sum;
	double running_sum_sq;
	double mean[32];   //shouldn't be more than 32 channels in a file?
	double std[32];
    unsigned char quantbyte;
    
    
    char *fitsdata = NULL;
    
    
    float nthr;
    float n_thr;
    
    float fitsval;
    unsigned int quantval;

    if(argc < 3) {
		fprintf(stderr, "USAGE: %s input.raw output.txt (use 'stdout' for output if stdout is desired)\n", argv[0]);
		exit(1);
	}
    
    float quantlookup[4];
    quantlookup[0] = 3.3358750;
    quantlookup[1] = 1.0;
    quantlookup[2] = -1.0;
    quantlookup[3] = -3.3358750;
    sprintf(pf.basefilename, argv[1]);

	sprintf(partfilename, argv[2]);

    pf.filenum=1;
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
           * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);





	
	fil = fopen(pf.basefilename, "rb");
	partfil = fopen(partfilename, "wb");	
	
	filepos=0;
	
	while(fread(buf, sizeof(char), 32768, fil)==32768) {		

		 fseek(fil, -32768, SEEK_CUR);
		 //printf("lhead: %d", lhead0);
		 fprintf(stderr, "length: %d\n", gethlength(buf));

		 guppi_read_obs_params(buf, &gf, &pf);
	 
		 fprintf(stderr, "nchan: %d\n", pf.hdr.nchan);    
		 fprintf(stderr, "size %d\n",pf.sub.bytes_per_subint + gethlength(buf));
		 by = by + pf.sub.bytes_per_subint + gethlength(buf);
		 fprintf(stderr, "mjd %Lf\n", pf.hdr.MJD_epoch);
		 fprintf(stderr, "zen: %f\n\n", pf.sub.tel_zen);
		 
		 fprintf(stderr, "packetindex %Ld\n", gf.packetindex);
		 fprintf(stderr, "packetsize: %d\n\n", gf.packetsize);
		 fprintf(stderr, "n_packets %d\n", gf.n_packets);
		 fprintf(stderr, "n_dropped: %d\n\n", gf.n_dropped);

		 if (pf.sub.data) {
		 	fprintf(stderr, "free pf.sub.data\n");
		 	fflush(stdout);
		 	free(pf.sub.data);         
         }
         pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);
		 
		 //need to allocate 4 bytes for each sample (float vals)
		 if (fitsdata) {
		 	fprintf(stderr, "free fitsdata\n");
			fflush(stdout);
		 	free(fitsdata);         
		 }
		 
		 /* allocate an array of floats for every sample in a sub integration */
		 fitsdata = (char *) malloc(pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits));
		 
		 fseek(fil, gethlength(buf), SEEK_CUR);
		 rv=fread(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, fil);		 
		 


		
		 if((long int)rv == pf.sub.bytes_per_subint){
			 fprintf(stderr, "%i\n", filepos);
			 fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));


			 if(filepos == 0) {
					 //first time through, output header
					 
					 //hputi4(buf, "NAXIS1", pf.sub.bytes_per_subint * (8/pf.hdr.nbits));
					 
					 fprintf(partfil, "Number of Channels: %d\n", pf.hdr.nchan);    
					 /* output channel center frequencies */
					 fprintf(partfil, "Start MJD: %Lf\n", pf.hdr.MJD_epoch);
					 fprintf(partfil, "Number of bits: %d\n", pf.hdr.nbits);
					 
					 printf("wrote: %d\n",(int) fwrite(buf, sizeof(char), gethlength(buf), partfil));  //write header
					 z=0;


					for(x=0;x < pf.sub.bytes_per_subint ;x=x+1) {
							//printf("%d\n", (int) ((signed char) pf.sub.data[x]));
						 	//printf("blah %d\n",z);
						 	//z=x+1;


						if(pf.hdr.nbits == 2){
							 for(a=0;a<4;a++){
								 quantval=0;
								 quantval = quantval + (pf.sub.data[x] >> (a * 2) & 1);
								 quantval = quantval + (2 * (pf.sub.data[x] >> (a * 2 + 1) & 1));
																 
								 
								 fitsval = quantlookup[quantval];

								 memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							     z = z + 4;									 
							 }
						} else {						
						 	fitsval = ((float) (signed char) pf.sub.data[x]) ;
						 	memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							z = z + 4;
						}	


					 	printf("%f\n", fitsval);							
						usleep(100000);

							//printf("%f\n", fitsval);
						 	//fitsval = (float) htonl((unsigned int) fitsval);
							//printf("%f\n", fitsval);							
					 		//usleep(1000000);
							
							//fwrite(&fitsval, sizeof(float), 1, partfil);
							
						 	//bin_print_verbose(fitsval);
						 	//fitsval = ((fitsval >> 8) & 0x00ff) | ((fitsval & 0x00ff) << 8);
						 	
					}
					
					
					
					printf("%d\n",x);
					imswap4(fitsdata,pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits));
					fwrite(fitsdata, sizeof(char), pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits), partfil);
					fflush(partfil);
					fclose(fil);
					fclose(partfil);
					exit(0);					 	
 			 }
		
		} else {
				fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		fprintf(stderr, "bytes: %d\n",by);
		fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	
	fclose(fil);
    exit(0);
}


void bin_print_verbose(short x)
/* function to print decimal numbers in verbose binary format */
/* x is integer to print, n_bits is number of bits to print */
{

   int j;
   printf("no. 0x%08x in binary \n",(int) x);

   for(j=16-1; j>=0;j--){
	   printf("bit: %i = %i\n",j, (x>>j) & 01);
   }

}
