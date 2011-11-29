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

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);


inline int quantize_2bit(struct psrfits *pf, double * mean, double * std);

int compute_stat(struct psrfits *pf, double *mean, double *std);

void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i 2bit.raw -o 8bit.raw ('-o stdout' allowed for output, -v or -V for verbose)\n", argv[0]);
}



/* 03/13 - edit to do optimized inplace quantization */

                                     
int main(int argc, char *argv[]) {
	struct guppi_params gf;
    struct psrfits pf;
    char buf[32768];
    char quantfilename[250]; //file name for dequantized file
    
	int filepos=0;
	size_t rv=0;
	int by=0;
    
    FILE *fil = NULL;   //input quantized file
    FILE *quantfil = NULL;  //dequantized file
    unsigned int quantval;
	unsigned int bitdepth;
	int x,y,z;
	int a,b,c;

	char twoeightlookup[4];
	twoeightlookup[0] = 40;
	twoeightlookup[1] = 12;
	twoeightlookup[2] = -12;
	twoeightlookup[3] = -40;


	
	double *mean = NULL;
	double *std = NULL;
	
    int vflag=0; //verbose



    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "Vvi:o:")) != -1)
         switch (c)
           {
           case 'v':
             vflag = 1;
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'i':
			 sprintf(pf.basefilename, optarg);
			 fil = fopen(pf.basefilename, "rb");
             break;
           case 'o':
			 sprintf(quantfilename, optarg);
			 if(strcmp(quantfilename, "stdout")==0) {
				 quantfil = stdout;
			 } else {
				 quantfil = fopen(quantfilename, "wb");			
			 }
             break;
           case '?':
             if (optopt == 'i' || optopt == 'o')
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


   

    pf.filenum=1;
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
           * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);


	

	if(!fil || !quantfil) {
		fprintf(stderr, "must specify input/output files\n");
		print_usage(argv);
		exit(1);
	}
	
	filepos=0;
	
	while(fread(buf, sizeof(char), 32768, fil)==32768) {		

		 fseek(fil, -32768, SEEK_CUR);

		 if(vflag>=1) fprintf(stderr, "length: %d\n", gethlength(buf));

		 guppi_read_obs_params(buf, &gf, &pf);
	 
		 if(vflag>=1) fprintf(stderr, "size %d\n",pf.sub.bytes_per_subint + gethlength(buf));
		 by = by + pf.sub.bytes_per_subint + gethlength(buf);
		 if(vflag>=1) fprintf(stderr, "mjd %Lf\n", pf.hdr.MJD_epoch);
		 if(vflag>=1) fprintf(stderr, "zen: %f\n\n", pf.sub.tel_zen);
		 if (pf.sub.data) free(pf.sub.data);
         pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);
		 
		 fseek(fil, gethlength(buf), SEEK_CUR);
		 rv=fread(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, fil);		 
		 


		
		 if((long int)rv == pf.sub.bytes_per_subint){
			 if(vflag>=1) fprintf(stderr, "%i\n", filepos);
			 if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));


					bitdepth = pf.hdr.nbits;
					/* update pf struct */
					pf.sub.bytes_per_subint = pf.sub.bytes_per_subint * 4;			
					pf.hdr.nbits = 8;			

					hputi4 (buf, "BLOCSIZE", pf.sub.bytes_per_subint);
					hputi4 (buf,"NBITS",pf.hdr.nbits);
				
					fwrite(buf, sizeof(char), gethlength(buf), quantfil);  //write header


					for(x=0;x < pf.sub.bytes_per_subint/4 ;x=x+1) {
//							printf("%d\n", (int) ((signed char) pf.sub.data[x]));
						 	//printf("blah %d\n",z);
						 	//z=x+1;


						if(bitdepth == 2){
							 for(a=0;a<4;a++){
								 quantval=0;
								 quantval = quantval + (pf.sub.data[x] >> (a * 2) & 1);
								 quantval = quantval + (2 * (pf.sub.data[x] >> (a * 2 + 1) & 1));
																 
								 //printf("%u\n", quantval);							
								 			/* bytes_per_subint now updated to be the proper length */
							     fwrite(&twoeightlookup[quantval], sizeof(char), 1, quantfil);  //write data

								 //fitsval = quantlookup[quantval];
								 //printf("%f\n", fitsval);							
								 //usleep(1000000);

								 //memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							     //z = z + 4;									 
								 //fprintf(stderr, "%u\n", quantval);
								 //usleep(1000000);
							 }
						} else {						
							fprintf(stderr, "This program only operates on 2bit data.\n");
							exit(0);
						 	//fitsval = ((float) (signed char) pf.sub.data[x]) ;
						 	//memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							//z = z + 4;
						}	
							//printf("%f\n", fitsval);
						 	//fitsval = (float) htonl((unsigned int) fitsval);
							//printf("%f\n", fitsval);							
					 		//usleep(1000000);
							
							//fwrite(&fitsval, sizeof(float), 1, partfil);
							
						 	//bin_print_verbose(fitsval);
						 	//fitsval = ((fitsval >> 8) & 0x00ff) | ((fitsval & 0x00ff) << 8);
						 	
					}


							filepos++;

			//quantize_2bit(&pf, mean, std);


			

			 
		} else {
				if(vflag>=1) fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		if(vflag>=1) fprintf(stderr, "bytes: %d\n",by);
		if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	



    //while ((rv=psrfits_read_subint(&pf))==0) { 
    //    printf("Read subint (file %d, row %d/%d)\n", 
    //            pf.filenum, pf.rownum-1, pf.rows_per_file);
    //}
    //if (rv) { fits_report_error(stderr, rv); }

	fclose(quantfil);
	fclose(fil);
    exit(0);
}





/* optimized 2-bit quantization */

/* applies 2 bit quantization to the data pointed to by pf->sub.data			*/
/* mean and std should be formatted as returned by 'compute_stat'				*/
/* quantization is performed 'in-place,' overwriting existing contents				*/
/* pf->hdr.nbits and pf->sub.bytes_per_subint are updated to reflect changes		*/
/* quantization scheme described at http://seti.berkeley.edu/kepler_seti_quantization  	*/

inline int quantize_2bit(struct psrfits *pf, double * mean, double * std) {

register unsigned int x,y;
unsigned int bytesperchan;




/* temporary variables for quantization routine */
float nthr[2];
float n_thr[2];
float chan_mean[2];
float sample;

register unsigned int offset;
register unsigned int address;

unsigned int pol0lookup[256];   /* Lookup tables for pols 0 and 1 */
unsigned int pol1lookup[256];


bytesperchan = pf->sub.bytes_per_subint/pf->hdr.nchan;

for(x=0;x < pf->hdr.nchan; x = x + 1)   {

		
		nthr[0] = (float) 0.98159883 * std[(x*pf->hdr.rcvr_polns) + 0];
		n_thr[0] = (float) -0.98159883 * std[(x*pf->hdr.rcvr_polns) + 0];
		chan_mean[0] = (float) mean[(x*pf->hdr.rcvr_polns) + 0];
		
		if(pf->hdr.rcvr_polns == 2) {
		   nthr[1] = (float) 0.98159883 * std[(x*pf->hdr.rcvr_polns) + 1];
		   n_thr[1] = (float) -0.98159883 * std[(x*pf->hdr.rcvr_polns) + 1];
		   chan_mean[1] = (float) mean[(x*pf->hdr.rcvr_polns) + 1];
		} else {
			nthr[1] = nthr[0];
			n_thr[1] = n_thr[0];
			chan_mean[1] = chan_mean[0];						
		}
								
		
		
		/* build the lookup table */
		for(y=0;y<128;y++) {   
			sample = ((float) y) - chan_mean[0]; 
			if (sample > nthr[0]) {
				pol0lookup[y] = 0;  						
			} else if (sample > 0) {
				pol0lookup[y] = 1; 												
			} else if (sample > n_thr[0]) {
				pol0lookup[y] = 2;																		 
			} else {
				pol0lookup[y] = 3;																		
			}	
		
			sample = ((float) y) - chan_mean[1]; 
			if (sample > nthr[1]) {
				pol1lookup[y] = 0;  						
			} else if (sample > 0) {
				pol1lookup[y] = 1; 												
			} else if (sample > n_thr[1]) {
				pol1lookup[y] = 2;																		 
			} else {
				pol1lookup[y] = 3;																		
			}			
		}
		
		for(y=128;y<256;y++) {   
			sample = ((float) y) - chan_mean[0] - 256; 
			if (sample > nthr[0]) {
				pol0lookup[y] = 0;  						
			} else if (sample > 0) {
				pol0lookup[y] = 1; 												
			} else if (sample > n_thr[0]) {
				pol0lookup[y] = 2;																		 
			} else {
				pol0lookup[y] = 3;																		
			}	
		
			sample = ((float) y) - chan_mean[1] - 256; 
			if (sample > nthr[1]) {
				pol1lookup[y] = 0;  						
			} else if (sample > 0) {
				pol1lookup[y] = 1; 												
			} else if (sample > n_thr[1]) {
				pol1lookup[y] = 2;																		 
			} else {
				pol1lookup[y] = 3;																		
			}			
		}


		/* memory position offset for this channel */
		offset = x * bytesperchan; 
		
		/* starting point for writing quantized data */
		address = offset/4;

		/* in this quantization code we'll sort-of assume that we always have two pols, but we'll set the pol0 thresholds to the pol1 values above if  */
		/* if only one pol is present. */
							
		for(y=0;y < bytesperchan; y = y + 4){
		
			/* form one 4-sample quantized byte */
			pf->sub.data[address] = pol0lookup[pf->sub.data[((offset) + y)]] + (pol0lookup[pf->sub.data[((offset) + y) + 1]] * 4) + (pol1lookup[pf->sub.data[((offset) + y) + 2]] * 16) + (pol1lookup[pf->sub.data[((offset) + y) + 3]] * 64);

			address++;																
		
		}					

}




return 1;
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



