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





/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);
double round(double x);

void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input_prefix -c channel -p N\n", argv[0]);
	fprintf(stderr, "		N = 2^N FFT Points\n");
	fprintf(stderr, "		-v or -V for verbose\n");
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
int overlap; // amount of overlap between sub integrations in samples

int N = 128;
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


	

 
	
	char filname[250];

	struct gpu_input rawinput;	
	
	long long int startindx;
	long long int curindx;
	int indxstep = 0;
	

	
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
    
	rawinput.file_prefix = NULL;
	rawinput.fil = NULL;
	rawinput.invalid = 0;
	rawinput.first_file_skip = 0;
	

    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "sVvi:o:")) != -1)
         switch (c)
           {
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
			 rawinput.file_prefix = optarg;
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

/* no input specified */
if(rawinput.file_prefix == NULL) {
	printf("WARNING no input stem specified%d\n", (i+1));
	exit(1);
}


/* set file counter to zero */
j = 0;

do {
	sprintf(filname, "%s.%04d.raw",rawinput.file_prefix,j);
	printf("%s\n",filname);		
	j++;
} while (exists(filname));
rawinput.filecnt = j-1;
printf("File count is %i\n",rawinput.filecnt);

/* didn't find any files */
if(rawinput.filecnt < 1) {
	printf("no files for stem %s found\n",rawinput.file_prefix);
	exit(1);		
}


/* open the first file for input */
sprintf(filname, "%s.0000.raw", rawinput.file_prefix);
rawinput.fil = fopen(filname, "rb");

printf("file open\n");


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

			
		   printf("packetindex %Ld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetindex %Ld\n", rawinput.gf.packetindex);
		   fprintf(stderr, "packetsize: %d\n\n", rawinput.gf.packetsize);
		   fprintf(stderr, "n_packets %d\n", rawinput.gf.n_packets);
		   fprintf(stderr, "n_dropped: %d\n\n",rawinput.gf.n_dropped);
		   fprintf(stderr, "bytes_per_subint: %d\n\n",rawinput.pf.sub.bytes_per_subint);

		   if (rawinput.pf.sub.data) free(rawinput.pf.sub.data);
		   
		   rawinput.pf.sub.data  = (unsigned char *) malloc(rawinput.pf.sub.bytes_per_subint);
			
	  } else {
	  		printf("couldn't read a header\n",rawinput.file_prefix);
			exit(1);
	  }
} else {
	printf("couldn't open first file\n",rawinput.file_prefix);
	exit(1);
}

	


//	tstart=band[first_good_band].pf.hdr.MJD_epoch;
//	tsamp = band[first_good_band].pf.hdr.dt * ftacc;

//	strcat(buf, strtok(band[first_good_band].pf.hdr.ra_str, ":"));
//	strcat(buf, strtok(band[first_good_band].pf.hdr.dec_str, ":"));

	


printf("calculating index step\n");
/* number of packets that we *should* increment by */
indxstep = (int) ((rawinput.pf.sub.bytes_per_subint * 4) / rawinput.gf.packetsize) - (int) (rawinput.overlap * rawinput.pf.hdr.nchan * rawinput.pf.hdr.rcvr_polns * 2 / rawinput.gf.packetsize);

//spectraperint = indxstep * band[first_good_band].gf.packetsize / (band[first_good_band].pf.hdr.nchan * band[first_good_band].pf.hdr.rcvr_polns * 2 * ftacc);

/* total bytes per channel, which is equal to total samples per channel, divided by number of samples summed in each filterbank integration */					
//spectraperint = ((rawinput.pf.sub.bytes_per_subint / rawinput.pf.hdr.nchan) - rawinput.overlap) / ftacc;	
overlap = rawinput.overlap;





printf("Index step: %d\n", indxstep);
printf("bytes per subint %d\n",rawinput.pf.sub.bytes_per_subint );
fflush(stdout);



//band[0].pf.sub.bytes_per_subint * (8 / band[0].pf.hdr.nbits) / (
//fastspectra = malloc(4 * 33046528) / (32 * 2 * 2)  == 1032704*/

//array = malloc(1000 * sizeof *array);


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
		  	  	exit(1);
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
							if(vflag>=1) fprintf(stderr,"read %d bytes from %d in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
						 	//load_spectra(fastspectra,  &rawinput.gf, &rawinput.pf, j);
						 } else {
						 	 rawinput.fil = NULL;
							 fprintf(stderr,"couldn't read as much as the header said we could... assuming corruption and exiting...\n");
							 exit(1);
						 }
						 
					
					} else if(rawinput.gf.packetindex > curindx) {
						 fprintf(stderr,"curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );

						 /* read a subint with too high an indx, must have dropped a whole subintegration*/
						 /* do nothing, will automatically have previous spectra (or 0) */



					} if(rawinput.gf.packetindex < curindx) {
						 fprintf(stderr,"curindx: %Ld, pktindx: %Ld\n", curindx, rawinput.gf.packetindex );

						 /* try to read an extra spectra ahead */
						 fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
						 rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 

						 if(fread(buf, sizeof(char), 32768, rawinput.fil) == 32768) {
							  
							  fseek(rawinput.fil, -32768, SEEK_CUR);
							  guppi_read_obs_params(buf, &rawinput.gf, &rawinput.pf);	 

							  if(rawinput.gf.packetindex == curindx) {
								  fseek(rawinput.fil, gethlength(buf), SEEK_CUR);
								  rv=fread(rawinput.pf.sub.data, sizeof(char), rawinput.pf.sub.bytes_per_subint, rawinput.fil);		 
				 
								  if((long int)rv == rawinput.pf.sub.bytes_per_subint){
									 fprintf(stderr,"read %d more bytes from %d in curfile %d\n", rawinput.pf.sub.bytes_per_subint, j, rawinput.curfile);
 	  						 	     //load_spectra(fastspectra,  &rawinput.gf, &rawinput.pf, j);
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

								

		
		


		fprintf(stderr, "dumping to disk\n");

		/* output one subint of accumulated spectra to filterbank file */

/*
		for(b=0;b<spectraperint;b++){
			for(a = fnchan-1; a > -1; a--) {	 			 	 
						   fwrite(&fastspectra[a][b],sizeof(float),1,partfil);			  
			 }
		}
*/
		if(rawinput.fil != NULL) curindx = curindx + indxstep;


} while(!(rawinput.invalid));
	
	
	fprintf(stderr, "finishing up...\n");

	if(vflag>=1) fprintf(stderr, "bytes: %ld\n",by);
	
	fclose(partfil);
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

int load_spectra(float **fastspectra, struct guppi_params *gf, struct psrfits *pf, int ofst)
{
int i,j,k,a;
int m=0;
float fitsval;	   
unsigned int quantval;
long int bytes_per_chan;
float samples[4096][4];


        
bytes_per_chan =  pf->sub.bytes_per_subint/pf->hdr.nchan;

/* buffer up one accumulation worth of complex samples */
/* either detect and accumulate, or fft, detect and accumulate */

	 for(j = (ofst * pf->hdr.nchan * N); j < ((ofst+1) * pf->hdr.nchan * N); j = j + N) {	 			 	 
		 for(k=0;k<spectraperint;k++){
			  
 			  for(i = 0; i < N; i++) fastspectra[j+i][k] = 0;

			  for(i = 0; i < ftacc;i++) {
				  /* sum FTACC dual pol complex samples */ 		
				  for(a=0;a<4;a++){

					   quantval=0;
					   quantval = quantval + (pf->sub.data[ (bytes_per_chan * m) + ( (k*ftacc) + i)] >> (a * 2) & 1);
					   quantval = quantval + (2 * (pf->sub.data[(bytes_per_chan * m)+ ( (k*ftacc) + i)] >> (a * 2 + 1) & 1));											 

					   //printf("%u\n", quantval);							
							  
					   //fitsval = quantlookup[quantval];
					   samples[i][a] = quantlookup[quantval];
					   
				  }
			  }	

		 	  for(a=0;a<ftacc;a=a+N){
					 for(i=0;i<N;i++){
						   in[i][0] = samples[i+a][0]; //real pol 0
						   in[i][1] = samples[i+a][1]; //imag pol 0
					 }	

					 fftwf_execute(p); /*do the FFT for pol 0*/  

					 
					 for(i=0;i<N;i++){
						 if(moment != 4) {						 
							  fastspectra[j+i][k] = fastspectra[j+i][k] + powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2);  /*real^2 + imag^2 for pol 0 */
						 } else {
							  fastspectra[j+i][k] = fastspectra[j+i][k] + powf((powf(out[(i+N/2)%N][1],2) + powf(out[(i+N/2)%N][1],2)),2);						 
						 }
					 }				
					 
					 for(i=0;i<N;i++){
						   in[i][0] = samples[i+a][2]; //real pol 0
						   in[i][1] = samples[i+a][3]; //imag pol 0
					 }	

					 fftwf_execute(p); /*do the FFT for pol 1*/  


					 for(i=0;i<N;i++){
						 if(moment != 4) {
							  fastspectra[j+i][k] = fastspectra[j+i][k] + powf(out[(i+N/2)%N][0],2) + powf(out[(i+N/2)%N][1],2);  /*real^2 + imag^2 for pol 1 */
						 } else {
							  fastspectra[j+i][k] = fastspectra[j+i][k] + powf((powf(out[(i+N/2)%N][1],2) + powf(out[(i+N/2)%N][1],2)),2);						 
						 }

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

