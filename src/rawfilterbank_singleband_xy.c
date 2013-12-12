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
#include "filterbank.h"
#include <fftw3.h>




int load_spectra(float **samples, float **fastspectra_xpol, float **fastspectra_ypol,float **fastspectra_xpolsq, float **fastspectra_ypolsq, struct guppi_params *gf, struct psrfits *pf, int ofst);

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);
double round(double x);

void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input.raw -o output.head ('-o stdout' allowed for output, -v or -V for verbose)\n", argv[0]);
}


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
int N = 8; //size of DFT on each GUPPI channel
int overlap; // amount of overlap between sub integrations in samples


int moment=2; //power to raise voltage to

fftwf_complex *in, *out;
fftwf_plan p;                 

int main(int argc, char *argv[]) {

	struct guppi_params gf;
    struct psrfits pf;
	int filecnt=0;
	int lastvalid=0;
    char buf[32768];
    char partfilename[250]; //file name for first part of file


	float **fastspectra_xpol;
	float **fastspectra_xpolsq;
	float **fastspectra_ypol;
	float **fastspectra_ypolsq;
	
	float** samples;


	
	char filname[250];

	struct gpu_input band[8];	
	
	long long int startindx;
	long long int curindx;
	int indxstep = 0;
	
	/* number of samples to sum in each spectra */

	/* samples per subint */
	/* (4 * 33046528) / (32 * 2 * 2)  == 1032704*/

	/* 128 * 3.2 * 10^-7s == 40.96 microseconds */
	ftacc = 128;
	
	quantlookup[0] = 3.3358750;
   	quantlookup[1] = 1.0;
   	quantlookup[2] = -1.0;
   	quantlookup[3] = -3.3358750;

    
    
	size_t rv=0;
	long unsigned int by=0;
    
    FILE *fil = NULL;   //input file
    FILE *xpolfil = NULL;  //output file for pol1
    FILE *ypolfil = NULL;  //output file for pol2
    FILE *xpolsqfil = NULL;  //output file for pol1
    FILE *ypolsqfil = NULL;  //output file for pol2
    
	int x,y,z;
	int a,b,c;
	int i,j,k;
    int vflag=0; //verbose
    
	for(i=0;i<8;i++) {band[i].file_prefix = NULL;band[i].fil = NULL;band[i].invalid = 0;band[i].first_file_skip = 0;}
	
	int first_good_band = -1;


    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "sVvi:o:1:2:3:4:5:6:7:8:N:")) != -1)
         switch (c)
           {
           case '1':
			 band[0].file_prefix = optarg;
             break;
           case '2':
			 band[1].file_prefix = optarg;
             break;
           case '3':
			 band[2].file_prefix = optarg;
             break;
           case '4':
			 band[3].file_prefix = optarg;
             break;
           case '5':
			 band[4].file_prefix = optarg;
             break;
           case '6':
			 band[5].file_prefix = optarg;
             break;
           case '7':
			 band[6].file_prefix = optarg;
             break;
           case '8':
			 band[7].file_prefix = optarg;
             break;
           case 'v':
             vflag = 1;
             break;
           case 's':
             moment = 4;
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'N':
			 N = atoi(optarg);
             break;
           case 'i':
			 sprintf(pf.basefilename, optarg);
			 fil = fopen(pf.basefilename, "rb");
             break;
           case 'o':
			 sprintf(partfilename, optarg);
			 if(strcmp(partfilename, "stdout")==0) {
				 xpolfil = stdout;
				 ypolfil = stdout;
			 } else {
				 sprintf(partfilename, "%s%s", optarg, ".xpol.fil");
				 xpolfil = fopen(partfilename, "wb");			

				 sprintf(partfilename, "%s%s", optarg, ".ypol.fil");
				 ypolfil = fopen(partfilename, "wb");			

				 sprintf(partfilename, "%s%s", optarg, ".xpolsq.fil");
				 xpolsqfil = fopen(partfilename, "wb");			

				 sprintf(partfilename, "%s%s", optarg, ".ypolsq.fil");
				 ypolsqfil = fopen(partfilename, "wb");			

			 }
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



           
	
	/* init these pointers so we don't crash if they're freed */
	
	pf.filenum=1;
    pf.sub.dat_freqs = (float *)malloc(sizeof(float));
    pf.sub.dat_weights = (float *)malloc(sizeof(float));
    pf.sub.dat_offsets = (float *)malloc(sizeof(float));
    pf.sub.dat_scales  = (float *)malloc(sizeof(float));
    pf.sub.data  = (unsigned char *)malloc(1);

	int numgoodbands=0;

	for(i=0;i<8;i++) {
		if(band[i].file_prefix == NULL) {
			printf("WARNING no input stem for band%d\n", (i+1));
			band[i].invalid = 1;
			//return 1;
		}
		j = 0;

		
		
		if(band[i].invalid != 1) {
            if(strstr(band[i].file_prefix, ".0000.raw") != NULL) memset(band[i].file_prefix + strlen(band[i].file_prefix) - 9, 0x0, 9);
            do {
				sprintf(filname, "%s.%04d.raw",band[i].file_prefix,j);
				printf("%s\n",filname);		
				j++;
			} while (exists(filname));
			band[i].filecnt = j-1;
			printf("%i\n",band[i].filecnt);
			if(band[i].filecnt < 1) {
				printf("no files for band %d\n",i);
				band[i].invalid = 1;
				
			}
	
	
			sprintf(filname, "%s.0000.raw", band[i].file_prefix);

			band[i].fil = fopen(filname, "rb");
			printf("file open for band %d\n", i);
		}
		
		if(band[i].fil){
			  if(fread(buf, sizeof(char), 32768, band[i].fil) == 32768){
				   
				   
				   guppi_read_obs_params(buf, &band[i].gf, &band[i].pf);
				   if(band[i].pf.hdr.nbits == 8) {
					  
					  fprintf(stderr, "got an 8bit header\n");
					  
					  /* figure out how the size of the first subint + header */
				      band[i].first_file_skip = band[i].pf.sub.bytes_per_subint + gethlength(buf);
						
					  /* rewind to the beginning */	
				   	  fseek(band[i].fil, -32768, SEEK_CUR);
				   	  
				   	  /* seek past the first subint + header */
				   	  fseek(band[i].fil, band[i].first_file_skip, SEEK_CUR);

					  /* read the next header */
				   	  fread(buf, sizeof(char), 32768, band[i].fil);
					  guppi_read_obs_params(buf, &band[i].gf, &band[i].pf);
					  fclose(band[i].fil);

				   } else {
					  //fseek(band[i].fil, -32768, SEEK_CUR);
		 
					  fclose(band[i].fil);
				   }
	
				   /* we'll use this band to set the params for the whole observation */
				   if(first_good_band == -1) first_good_band = i;
					
				   lastvalid = i;
				   printf("Band %d valid\n", i);
	  			   numgoodbands++;

	  			   band[i].fil = NULL;

				   hgeti4(buf, "OVERLAP", &band[i].overlap);

				   //band[i].gf = gf;
				   //band[i].pf = pf;
				   //fclose(fil);
					
				 printf("BAND: %d packetindex %Ld\n", i, band[i].gf.packetindex);
				   fprintf(stderr, "packetindex %Ld\n", band[i].gf.packetindex);
				   fprintf(stderr, "packetsize: %d\n\n", band[i].gf.packetsize);
				   fprintf(stderr, "n_packets %d\n", band[i].gf.n_packets);
				   fprintf(stderr, "n_dropped: %d\n\n",band[i].gf.n_dropped);
				   
				   if (band[i].pf.sub.data) free(band[i].pf.sub.data);
				   band[i].pf.sub.data  = (unsigned char *) malloc(band[i].pf.sub.bytes_per_subint);
	  
				   /* Check to see if the current band matches the previous */
				   if(i > 0) {
					   if( band[i].gf.packetindex != band[lastvalid].gf.packetindex ) {fprintf(stderr, "packetindex mismatch\n"); return 1;}
					   if( band[i].filecnt != band[lastvalid].filecnt ) {fprintf(stderr, "filecnt mismatch\n"); return 1;}
					   if( band[i].pf.hdr.fctr < band[lastvalid].pf.hdr.fctr ) {fprintf(stderr, "freq misordered\n"); return 1;}
				   }			
			  }
		}
	}
	

	
	
	int fnchan=0;
	if(numgoodbands < 2) {
		fnchan = band[first_good_band].pf.hdr.nchan * N;
	
	} else {
	  
		  /* total number of channels in the filterbank - 256 channelized by guppi * second SW stage */
		  /* really should pull this out of the .raw files */	
		fnchan = band[first_good_band].pf.hdr.nchan * N * 8;
	}
	
	
	/* take important values from first good band */
	
	/* number of packets that we *should* increment by */
	indxstep = (int) ((band[first_good_band].pf.sub.bytes_per_subint * 4) / band[first_good_band].gf.packetsize) - (int) (band[first_good_band].overlap * band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 / band[first_good_band].gf.packetsize);
	
	//spectraperint = indxstep * band[first_good_band].gf.packetsize / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 * ftacc);
	
	/* total bytes per channel, which is equal to total samples per channel, divided by number of samples summed in each filterbank integration */					
	spectraperint = ((band[first_good_band].pf.sub.bytes_per_subint / band[first_good_band].pf.hdr.nchan) - band[first_good_band].overlap) / ftacc;	

	printf("spectra per subint%f\n", (((float) band[first_good_band].pf.sub.bytes_per_subint / (float) band[first_good_band].pf.hdr.nchan) - (float) band[first_good_band].overlap) / (float) ftacc);
	overlap = band[first_good_band].overlap;
	
	
	fastspectra_xpol = (float**) malloc(fnchan * sizeof(float*));  
	fastspectra_ypol = (float**) malloc(fnchan * sizeof(float*));  
	fastspectra_xpolsq = (float**) malloc(fnchan * sizeof(float*));  
	fastspectra_ypolsq = (float**) malloc(fnchan * sizeof(float*));  
	
	for (i = 0; i < fnchan; i++) fastspectra_xpol[i] = (float*) malloc(spectraperint*sizeof(float));
	for (i = 0; i < fnchan; i++) fastspectra_ypol[i] = (float*) malloc(spectraperint*sizeof(float));
	for (i = 0; i < fnchan; i++) fastspectra_xpolsq[i] = (float*) malloc(spectraperint*sizeof(float));
	for (i = 0; i < fnchan; i++) fastspectra_ypolsq[i] = (float*) malloc(spectraperint*sizeof(float));

	 
	/* init fast spectra array (one subinterval)*/
	for(i = 0; i < fnchan; i++) {
		  for(j = 0; j < spectraperint;j++) {
		  fastspectra_xpol[i][j] = 0.0;
		  fastspectra_ypol[i][j] = 0.0;
		  fastspectra_xpolsq[i][j] = 0.0;
		  fastspectra_ypolsq[i][j] = 0.0;

		  }	
	}



	printf("spectra array malloc'd\n");
 /* 
 Need to set this filterbank stuff before dumping filterbank header 
hdr->BW  
  //source_name
*/
	machine_id=0;
	telescope_id=6;
	data_type=1;
	nbits=32;
	obits=32;
	nbeams = 0;
	ibeam = 0;
	tstart=band[first_good_band].pf.hdr.MJD_epoch;
	tsamp = band[first_good_band].pf.hdr.dt * ftacc;
	nifs = 1;
	src_raj=0.0;
	src_dej=0.0;
	az_start=0.0;
	za_start=0.0;
    strcpy(ifstream,"YYYY");
	nchans = fnchan;


	memset(buf, 0x0, 32768);
	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
	strcat(buf, strtok((char *) 0, ":"));
	strcat(buf, strtok((char *) 0, ":"));
	src_raj = strtod(buf, (char **) NULL);
	
	memset(buf, 0x0, 32768);
	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));
	strcat(buf, strtok((char *) 0, ":"));
	strcat(buf, strtok((char *) 0, ":"));
	src_dej = strtod(buf, (char **) NULL);

	printf("filterbank vars set\n");

	if(numgoodbands < 2) {

		  foff = band[first_good_band].pf.hdr.BW / fnchan * -1.0;
		  fch1= band[first_good_band].pf.hdr.fctr - ((fnchan/2)-0.5)*foff;
	
	} else {
	  
		  foff = (band[first_good_band].pf.hdr.BW * 8) / fnchan * -1.0;
		  fch1= 1500.0 - ((fnchan/2)-0.5)*foff;
	}
	

		printf("set freqs\n");


	/* dump filterbank header */
	filterbank_header(xpolfil);
	filterbank_header(ypolfil);
	filterbank_header(xpolsqfil);
	filterbank_header(ypolsqfil);
	
		printf("dumped header\n");

	in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
	out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
	p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_PATIENT);

	
	printf("fft planned\n");

	

	
samples = (float**) malloc(ftacc*sizeof(float*));
for (i = 0; i < ftacc; i++)
   samples[i] = (float*) malloc(4*sizeof(float));

for (i = 0; i < ftacc; i++) {
samples[i][0] = 0.0;
samples[i][1] = 0.0;
samples[i][2] = 0.0;
samples[i][3] = 0.0;
samples[i][3] = samples[i][0];
}

	printf("Index step: %d\n", indxstep);
	printf("bytes per subint %d\n",band[first_good_band].pf.sub.bytes_per_subint );
	fflush(stdout);


	

	//band[0].pf.sub.bytes_per_subint * (8 / band[0].pf.hdr.nbits) / (
	//fastspectra = malloc(4 * 33046528) / (32 * 2 * 2)  == 1032704*/

	//array = malloc(1000 * sizeof *array);


	startindx = band[first_good_band].gf.packetindex;
	curindx = startindx;
	int readvalid=0;

	filecnt = band[first_good_band].filecnt;

	for(i=0;i<8;i++) band[i].curfile = 0;			

	do{
	/* for now we'll assume that gpu1 has the first valid packetindex */	

		for(j=0;j<8;j++){										
			 if(!band[j].invalid){						  
				  if(band[j].fil == NULL) {
					  /* no file is open for this band, try to open one */
					  sprintf(filname, "%s.%04d.raw",band[j].file_prefix,band[j].curfile);
					  printf("filename is %s\n",filname);
					  if(exists(filname)){
						 printf("opening %s\n",filname);				
						 band[j].fil = fopen(filname, "rb");			 
						 if(band[j].curfile == 0 && band[j].first_file_skip != 0) fseek(band[j].fil, band[j].first_file_skip, SEEK_CUR);  
					  }	else band[j].invalid = 1;
				  }

				  if(band[j].fil){
						if(fread(buf, sizeof(char), 32768, band[j].fil) == 32768) {
							readvalid=1;

							fseek(band[j].fil, -32768, SEEK_CUR);
							if(vflag>=1) fprintf(stderr, "band: %d length: %d\n", j, gethlength(buf));
							guppi_read_obs_params(buf, &band[j].gf, &band[j].pf);
							if(vflag>=1) {
								 fprintf(stderr, "BAND: %d packetindex %Ld\n", j, band[j].gf.packetindex);
								 fprintf(stderr, "packetindex %Ld\n", band[j].gf.packetindex);
								 fprintf(stderr, "packetsize: %d\n\n", band[j].gf.packetsize);
								 fprintf(stderr, "n_packets %d\n", band[j].gf.n_packets);
								 fprintf(stderr, "n_dropped: %d\n\n",band[j].gf.n_dropped);
							}
							
					   		if(band[j].gf.packetindex == curindx) {
								 /* read a subint with correct index */
								 fseek(band[j].fil, gethlength(buf), SEEK_CUR);
								 rv=fread(band[j].pf.sub.data, sizeof(char), band[j].pf.sub.bytes_per_subint, band[j].fil);		 
				
								 if((long int)rv == band[j].pf.sub.bytes_per_subint){
									if(vflag>=1) fprintf(stderr,"read %d bytes from %d in curfile %d\n", band[j].pf.sub.bytes_per_subint, j, band[j].curfile);
								 	if(numgoodbands < 2) {
									 	load_spectra(samples, fastspectra_xpol, fastspectra_ypol, fastspectra_xpolsq, fastspectra_ypolsq,  &band[j].gf, &band[j].pf, 0);
									} else {
									 	load_spectra(samples, fastspectra_xpol, fastspectra_ypol, fastspectra_xpolsq, fastspectra_ypolsq,  &band[j].gf, &band[j].pf, j);									
									}
								 } else {
									 fprintf(stderr,"something went wrong.. couldn't read as much as the header said we could...\n");
								 }
								 
							
							} else if(band[j].gf.packetindex > curindx) {
								 fprintf(stderr,"curindx: %Ld, pktindx: %Ld\n", curindx, band[j].gf.packetindex );
								 /* read a subint with too high an indx, must have skipped a packet*/
								 /* do nothing, will automatically have previous spectra (or 0) */
							} if(band[j].gf.packetindex < curindx) {
								 fprintf(stderr,"curindx: %Ld, pktindx: %Ld\n", curindx, band[j].gf.packetindex );

								 /* try to read an extra spectra ahead */
								 fseek(band[j].fil, gethlength(buf), SEEK_CUR);
								 rv=fread(band[j].pf.sub.data, sizeof(char), band[j].pf.sub.bytes_per_subint, band[j].fil);		 

								 if(fread(buf, sizeof(char), 32768, band[j].fil) == 32768) {
		 							  
		 							  fseek(band[j].fil, -32768, SEEK_CUR);
									  guppi_read_obs_params(buf, &band[j].gf, &band[j].pf);	 

									  if(band[j].gf.packetindex == curindx) {
										  fseek(band[j].fil, gethlength(buf), SEEK_CUR);
										  rv=fread(band[j].pf.sub.data, sizeof(char), band[j].pf.sub.bytes_per_subint, band[j].fil);		 
						 
										  if((long int)rv == band[j].pf.sub.bytes_per_subint){
											 fprintf(stderr,"read %d more bytes from %d in curfile %d\n", band[j].pf.sub.bytes_per_subint, j, band[j].curfile);

											 if(numgoodbands < 2) {
												 load_spectra(samples, fastspectra_xpol, fastspectra_ypol, fastspectra_xpolsq, fastspectra_ypolsq,  &band[j].gf, &band[j].pf, 0);
											 } else {
												 load_spectra(samples, fastspectra_xpol, fastspectra_ypol, fastspectra_xpolsq, fastspectra_ypolsq,  &band[j].gf, &band[j].pf, j);									
											 }

										  } else {
										  	 fprintf(stderr,"something went wrong.. couldn't read as much as the header said we could...\n");
										  }

										} else {
										  fprintf(stderr,"skipped an extra packet, still off\n");
										  /* do nothing, will automatically have previous spectra (or 0) */
										  /* this shouldn't happen very often....  */											
										}
								 } else {
									  /* file open but couldn't read 32KB */
									  fclose(band[j].fil);
									  band[j].fil = NULL;
									  band[j].curfile++;														 								 
								 }								 
							
							}

						} else {

						/* file open but couldn't read 32KB */
						   fclose(band[j].fil);
						   band[j].fil = NULL;
						   band[j].curfile++;						
						}
				  }			 	 	 
			 }

								
		}
		
		

		if(readvalid == 1) {

		fprintf(stderr, "dumping to disk\n");

		fflush(stderr);
		/* output one subint of accumulated spectra to filterbank file */
		for(b=0;b<spectraperint;b++){
			for(a = fnchan-1; a > -1; a--) {	 			 	 
						   fwrite(&fastspectra_xpol[a][b],sizeof(float),1,xpolfil);			  
						   fwrite(&fastspectra_xpolsq[a][b],sizeof(float),1,xpolsqfil);			  
						   fwrite(&fastspectra_ypol[a][b],sizeof(float),1,ypolfil);			  
						   fwrite(&fastspectra_ypolsq[a][b],sizeof(float),1,ypolsqfil);			  	
			}
		}


		//if (band[0].curfile == 1) {
		//	fclose(partfil);
		//	return 1;		
		//}
		
		/* made it through all 8 bands, now increment current pkt index */
		curindx = curindx + indxstep;
		readvalid = 0;
		fflush(stderr);
		}
	} while(!(band[0].invalid && band[1].invalid && band[2].invalid && band[3].invalid && band[4].invalid && band[5].invalid && band[6].invalid && band[7].invalid));
	
	

	fprintf(stderr, "finishing up...\n");

	for (i = 0; i < ftacc; i++){
	   free(samples[i]);
	}
	free(samples);

	if(vflag>=1) fprintf(stderr, "bytes: %ld\n",by);
	//if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	fclose(xpolfil);
	fclose(ypolfil);
	fclose(xpolsqfil);
	fclose(ypolsqfil);
	
	
	fprintf(stderr, "closed output file...\n");

	fftwf_destroy_plan(p);
	fftwf_free(in); 
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



/* spectraperint is the number of accumulations in each sub interval */

int load_spectra(float **samples, float **fastspectra_xpol, float **fastspectra_ypol, float **fastspectra_xpolsq, float **fastspectra_ypolsq, struct guppi_params *gf, struct psrfits *pf, int ofst)
{
int i,j,k,a;
int m=0;
unsigned int quantval;
long int bytes_per_chan;
//float samples[5000000][4];
float tempfloat;



bytes_per_chan =  pf->sub.bytes_per_subint/pf->hdr.nchan;

/* buffer up one accumulation worth of complex samples */
/* either detect and accumulate, or fft, detect and accumulate */

	 for(j = (ofst * pf->hdr.nchan * N); j < ((ofst+1) * pf->hdr.nchan * N); j = j + N) {	 			 	 
		 for(k=0;k<spectraperint;k++){
			  
 			  for(i = 0; i < N; i++) fastspectra_xpol[j+i][k] = 0;
 			  for(i = 0; i < N; i++) fastspectra_ypol[j+i][k] = 0;
 			  for(i = 0; i < N; i++) fastspectra_xpolsq[j+i][k] = 0;
 			  for(i = 0; i < N; i++) fastspectra_ypolsq[j+i][k] = 0;

			  for(i = 0; i < ftacc;i++) {
				  /* sum FTACC dual pol complex samples */ 		
				  for(a=0;a<4;a++){

					   //quantval=0;
					   quantval = (pf->sub.data[ (bytes_per_chan * m) + ( (k*ftacc) + i)] >> (a * 2) & 1);
					   quantval = quantval + (2 * (pf->sub.data[(bytes_per_chan * m)+ ( (k*ftacc) + i)] >> (a * 2 + 1) & 1));											 

					   //printf("%u\n", quantval);							
							  
					   //fitsval = quantlookup[quantval];
					   samples[i][a] = quantlookup[quantval];
					   
				  }
			  }	

			  if(N > 1) {
				  for(a=0;a<ftacc;a=a+N){
						 
						 for(i=0;i<N;i++){
							   in[i][0] = samples[i+a][0]; //real pol 0
							   in[i][1] = samples[i+a][1]; //imag pol 0
						 }	
	
						 fftwf_execute(p); /*do the FFT for pol 0*/  
						 
						 for(i=0;i<N;i++){
								  fastspectra_xpol[j+i][k] = fastspectra_xpol[j+i][k] + powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2);  /*real^2 + imag^2 for pol 0 */
								  fastspectra_xpolsq[j+i][k] = fastspectra_xpolsq[j+i][k] + powf((powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2)),2);						 
						 }				
						 
						 for(i=0;i<N;i++){
							   in[i][0] = samples[i+a][2]; //real pol 0
							   in[i][1] = samples[i+a][3]; //imag pol 0
						 }	
	
						 fftwf_execute(p); /*do the FFT for pol 1*/  
	
						 for(i=0;i<N;i++){
								  fastspectra_ypol[j+i][k] = fastspectra_ypol[j+i][k] + powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2);  /*real^2 + imag^2 for pol 1 */
								  fastspectra_ypolsq[j+i][k] = fastspectra_ypolsq[j+i][k] + powf((powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2)),2);						 
						 }						 
	
				  }			 	  	
			  } else {
			  
				  for(a=0;a<ftacc;a=a+1){
						 
						//samples[a][0]; //real pol 0
						//samples[a][1]; //imag pol 0
							 
						tempfloat =  powf(samples[a][0],2) + powf(samples[a][1],2);	 
						fastspectra_xpol[j][k] = fastspectra_xpol[j][k] + tempfloat;  /*real^2 + imag^2 for pol 0 */
						fastspectra_xpolsq[j][k] = fastspectra_xpolsq[j][k] + powf(tempfloat,2);						 
						 
						//samples[a][2]; //real pol 0
						//samples[a][3]; //imag pol 0
						tempfloat = powf(samples[a][2],2) + powf(samples[a][3],2);
						fastspectra_ypol[j][k] = fastspectra_ypol[j][k] + tempfloat;  /*real^2 + imag^2 for pol 1 */
						fastspectra_ypolsq[j][k] = fastspectra_ypolsq[j][k] + powf(tempfloat,2);						 
	
				  }				  
			  
			  }
		
			  
		 }
		 m++;
	 }






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


#include "filterbank.h"

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
  fprintf(stderr,"the code if necessary... Contact dunc@naic.edu for further help\n");
  exit(0);
}

int big_endian() /*includefile*/
{
  return (!little_endian());
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

