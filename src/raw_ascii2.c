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


/* samples per subint */
/* (4 * 33046528) / (32 * 2 * 2)  == 1032704*/
/* 128 * 3.2 * 10^-7s == 40.96 microseconds */


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
};

float quantlookup[4];
int ftacc;
int spectraperint;
long int bytesperchan;
long int overlap_bytes;
int nchan;
int moment=2; //power to raise voltage to

              

int main(int argc, char *argv[]) {

	struct guppi_params gf;
    struct psrfits pf;
	int filecnt=0;
	int lastvalid=0;
    char buf[32768];
    char partfilename[250]; //file name for first part of file


	float **fastspectra;
	
	/* total number of channels in the filterbank - 256 channelized by guppi * second SW stage */
	/* really should pull this out of the .raw files */
	
	int fnchan = 256;  
	
	char filname[250];

	struct gpu_input band[8];	
	
	long long int startindx;
	long long int curindx;
	int indxstep = 0;
	
	ftacc = 128;
	quantlookup[0] = 3.3358750;
   	quantlookup[1] = 1.0;
   	quantlookup[2] = -1.0;
   	quantlookup[3] = -3.3358750;

    
    
	int filepos=0;
	size_t rv=0;
	long unsigned int by=0;
    
    FILE *fil = NULL;   //input file
    FILE *partfil = NULL;  //partial file
    
	int x,y,z;
	int a,b,c;
	int i,j,k;
    int vflag=0; //verbose
    
	for(i=0;i<8;i++) {band[i].file_prefix = NULL;band[i].fil = NULL;band[i].invalid = 0;}
	
	int first_good_band = -1;


	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "sVvi:o:1:2:3:4:5:6:7:8:")) != -1)
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
           case 'i':
			 sprintf(pf.basefilename, optarg);
			 fil = fopen(pf.basefilename, "rb");
             break;
           case 'o':
			 sprintf(partfilename, optarg);
			 if(strcmp(partfilename, "stdout")==0) {
				 partfil = stdout;
			 } else {
				 partfil = fopen(partfilename, "wb");			
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

	for(i=0;i<8;i++) {
		if(band[i].file_prefix == NULL) {
			printf("WARNING no input stem for band%d\n", (i+1));
			band[i].invalid = 1;
			//return 1;
		}
		j = 0;

		
		
		if(band[i].invalid != 1) {
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
			printf("file open %d\n" , band[i].fil);
		}
		
		if(band[i].fil){
			  if(fread(buf, sizeof(char), 32768, band[i].fil) == 32768){
				   
				   /* we'll use this band to set the params for the whole observation */
				   if(first_good_band == -1) first_good_band = i;
					
				   lastvalid = i;
				   printf("Band %d valid\n", i);
	  			   
				   fseek(band[i].fil, -32768, SEEK_CUR);
	  
				   fclose(band[i].fil);
	  			   band[i].fil = NULL;
				   guppi_read_obs_params(buf, &band[i].gf, &band[i].pf);
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
	
 /* 
 Need to set this filterbank stuff before dumping filterbank header 
  
  //source_name
*/
	machine_id=0;
	telescope_id=6;
	data_type=1;
	foff = 800.00 / fnchan * -1.0;
	fch1= 1500.0 - ((fnchan/2)-0.5)*foff;
	nchans = fnchan;
	nbeams = 0;
	ibeam = 0;
	nbits=32;
	obits=32;
	tstart=band[first_good_band].pf.hdr.MJD_epoch;
	tsamp = band[first_good_band].pf.hdr.dt * ftacc;
	nifs = 1;
	src_raj=0.0;
	src_dej=0.0;
	az_start=0.0;
	za_start=0.0;
    strcpy(ifstream,"YYYY");

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


	
	/* take important values from the first good band */
	
	
	indxstep = (int) ((band[first_good_band].pf.sub.bytes_per_subint * 4) / band[first_good_band].gf.packetsize) - (int) (band[first_good_band].overlap * band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 / band[first_good_band].gf.packetsize);
	
	//the band[n].overlap value is in samples, but we get 4 values for each sample (2 pols, real and imaginary)
	overlap_bytes = band[first_good_band].overlap;
		
	//spectraperint = indxstep * band[first_good_band].gf.packetsize / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 * ftacc);
	/* number of bytes per quantized frequency channel */
    //bytesperchan = (band[first_good_band].pf.sub.bytes_per_subint) / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2);
	bytesperchan = (band[first_good_band].pf.sub.bytes_per_subint) / (band[first_good_band].pf.hdr.nchan);
    
    nchan = band[first_good_band].pf.hdr.nchan;

	printf("Index step: %d\n", indxstep);
	printf("bytes per subint is %d\n", band[first_good_band].pf.sub.bytes_per_subint);
	
	fflush(stdout);

	fastspectra = (float**) malloc(fnchan * sizeof(float*));  
	for (i = 0; i < fnchan; i++) fastspectra[i] = (float*) malloc(spectraperint*sizeof(float));
	

	//band[0].pf.sub.bytes_per_subint * (8 / band[0].pf.hdr.nbits) / (
	//fastspectra = malloc(4 * 33046528) / (32 * 2 * 2)  == 1032704*/

	//array = malloc(1000 * sizeof *array);


	startindx = band[first_good_band].gf.packetindex;
	curindx = startindx;
	
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
					  }	else band[j].invalid = 1;
				  }
				  //printf("and here\n");
				  if(band[j].fil){
						if(fread(buf, sizeof(char), 32768, band[j].fil) == 32768) {
						
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
								 	
								 	load_spectra(partfil,  &band[j].gf, &band[j].pf, j);
								 	
								 	
								 } else {
									 fprintf(stderr,"something went wrong.. couldn't read as much as the header said we could...\n");
								 }
								 
							
							} else if(band[j].gf.packetindex > curindx) {
								 fprintf(stderr,"curindx: %Ld, pktindx: %Ld\n", curindx, band[j].gf.packetindex );


								 /* read a subint with too high an indx, must have dropped an entire subint*/
								 /* output a subint full of zeros */


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
 	  								 	     
 	  								 	     load_spectra(partfil,  &band[j].gf, &band[j].pf, j);
 	  								 	     
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

			 if(band[j].invalid){
			 /*  output a subint worth of zeroes */
			 	load_spectra(partfil,  NULL, NULL, j);
			 }
		}
		
		


		fprintf(stderr, "dumping to disk\n");

		/* output one subint of accumulated spectra to filterbank file */
		for(b=0;b<spectraperint;b++){
			for(a = fnchan-1; a > -1; a--) {	 			 	 
						   //fwrite(&fastspectra[a][b],sizeof(float),1,partfil);			  
			 }
		}


		//if (band[0].curfile == 1) {
		//	fclose(partfil);
		//	return 1;		
		//}
		
		/* made it through all 8 bands, now increment current pkt index */
		curindx = curindx + indxstep;

	} while(!(band[0].invalid && band[1].invalid && band[2].invalid && band[3].invalid && band[4].invalid && band[5].invalid && band[6].invalid && band[7].invalid));
	
	

	fprintf(stderr, "finishing up...\n");

	if(vflag>=1) fprintf(stderr, "bytes: %ld\n",by);
	if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	fclose(partfil);
	fprintf(stderr, "closed output file...\n");


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

int load_spectra(FILE *partfil, struct guppi_params *gf, struct psrfits *pf, int ofst)
{
int i,j,k,a;
int m=0;
float fitsval;	   
unsigned int quantval;
float samples[4096][4];
long int samples_to_dump;
float power;
samples_to_dump = bytesperchan;

samples_to_dump = 40;


         
/* buffer up one accumulation worth of complex samples */
/* either detect and accumulate, or fft, detect and accumulate */


	
	//fprintf(partfil, "channel: %d %d\n", ofst, nchan);
	if(pf != NULL) {
		  for(i=0;i < nchan;i++){
			  //fprintf(partfil, "channel: %d\n", (ofst * nchan) + i);
			  for(j=0;j<samples_to_dump;j++) {	 			 	 
	  					power = 0;
						/* sum FTACC dual pol complex samples */ 		
						for(a=0;a<4;a++){
	  			
							 quantval=0;
							 quantval = quantval + (pf->sub.data[ (i*bytesperchan) + j] >> (a * 2) & 1);
							 quantval = quantval + (2 * (pf->sub.data[ (i*bytesperchan) + j] >> (a * 2 + 1) & 1));											 
							 power = power + powf(quantlookup[quantval] ,2);	  
							 //fprintf(partfil, "%f,", quantlookup[quantval]);														  					   
						}
					    fprintf(partfil, "%f,", power);														  					   

					  
			   }		 
			  
			  fprintf(partfil, "\n");														  					   
		   }
	} else {
		  
		  for(i=0;i < nchan;i++){
			  fprintf(partfil, "channel: %d\n", (ofst * nchan) + i);
			  for(j=0;j<bytesperchan;j++) {	 			 	 	
						/* sum FTACC dual pol complex samples */ 		
						for(a=0;a<4;a++){	  	  
							 fprintf(partfil, "0.0,");														  					   
						}					  
			   }		 
			  
			  fprintf(partfil, "\n");														  					   
		   }	
	
	
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

