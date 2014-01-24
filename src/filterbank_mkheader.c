/* generate a filterbank-format header for appending to an existing time/frequency dataset */
/* Example: */
/*  ./filterbank_header -o blah -nifs 1 -fch1 1420 -source B0329+54 -filename foobar.bin */
/* -telescope ARECIBO -src_raj 032900 -src_dej 540000 -tsamp 65 -foff -0.25 -nbits 8  */
/* -nchans 1024 -tstart 53543.23 */
/* filterbank-format data should be in descending frequency order, and foff should be < 0 */
/* 8-bit data are interpreted as unsigned integers, 32-bit data are interpreted as IEEE floats */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) {
    return 1;
  } else {
    return 0;
  }
}
void send_string(char *string, FILE *output) /* includefile */
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
  /*fprintf(stderr,"%s\n",string);*/
}

void send_float(char *name,float floating_point, FILE *output) /* includefile */
{
  send_string(name, output);
  fwrite(&floating_point,sizeof(float),1,output);
  /*fprintf(stderr,"%f\n",floating_point);*/
}

void send_double (char *name, double double_precision, FILE *output) /* includefile */
{
  send_string(name, output);
  fwrite(&double_precision,sizeof(double),1,output);
  /*fprintf(stderr,"%f\n",double_precision);*/
}

void send_int(char *name, int integer, FILE *output) /* includefile */
{
  send_string(name, output);
  fwrite(&integer,sizeof(int),1,output);
  /*fprintf(stderr,"%d\n",integer);*/
}

void send_char(char *name, char integer, FILE *output) /* includefile */
{
  send_string(name, output);
  fwrite(&integer,sizeof(char),1,output);
}


void send_long(char *name, long integer, FILE *output) /* includefile */
{
  send_string(name, output);
  fwrite(&integer,sizeof(long),1,output);
  /*fprintf(stderr,"%ld\n",integer);*/
}

void send_coords(double raj, double dej, double az, double za, FILE *output) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj, output);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej, output);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az, output);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za, output);
}



int main (int argc, char *argv[]) {

long int i=1,j=0,k=0;
FILE *output;

char filename[250];
char rawfilename[250];

char telescope[250];

int telescope_id;
int data_type;
char source_name[250];
double fch1;
int nchans;
double foff;
int nifs;
double tsamp;
double src_raj;
double src_dej;
double tstart;
int nbits;
int nbeams=1;
int ibeam=1;
int machine_id = -1;
double az_start=0.0;
double za_start=0.0;



fprintf(stderr,"...\n");
while (i<argc) {
    
    if (strings_equal(argv[i],"-o")) {
	/* get and open file for output */
	   strcpy(filename,argv[++i]);
	   output=fopen(filename,"wb");
    } else if (strings_equal(argv[i],"-filename")) {
		 /* get name of raw input file */
		  strcpy(rawfilename,argv[++i]);
	} else if (strings_equal(argv[i],"-source")) {
		 /* get name of raw input file */
		  strcpy(source_name,argv[++i]);
	} else if (strings_equal(argv[i],"-telescope")) {
		 /* get name of raw input file */
		  strcpy(telescope,argv[++i]);		   
    } else if (strings_equal(argv[i],"-tstart")) {
		 /* get starting time (s) */
		 tstart=atof(argv[++i]);
	} else if (strings_equal(argv[i],"-tsamp")) {
		 i++;
		 tsamp=1.0e-6*atof(argv[i]);
	} else if (strings_equal(argv[i],"-fch1")) {
		 i++;
		 fch1=atof(argv[i]);
	} else if (strings_equal(argv[i],"-src_raj")) {
		 i++;
		 src_raj=atof(argv[i]);
	} else if (strings_equal(argv[i],"-src_dej")) {
		 i++;
		 src_dej=atof(argv[i]);		 		 
    } else if (strings_equal(argv[i],"-foff")) {
		 i++;
		 foff=atof(argv[i]);
    } else if (strings_equal(argv[i],"-nbits")) {
		 /* output number of bits per sample to write */
		 nbits=atoi(argv[++i]);
    } else if (strings_equal(argv[i],"-nchans")) {
		 /* output number of channels */
		 nchans=atoi(argv[++i]);		 
	 } else if (strings_equal(argv[i],"-nifs")) {
		 i++;
		 nifs=atoi(argv[i]);
      } else {
	/* unknown argument passed down - stop! */
	//filterbank_help();
	fprintf(stderr,"Unknown argument (%s) passed to filterbank\n",argv[i]);
	exit(1);
      }
      i++;
    }
  

  if (strcasecmp(telescope,"PARKES")==0)
    telescope_id=4;
  else if (strcasecmp(telescope,"ARECIBO")==0)
    telescope_id=1;
  else if (strcasecmp(telescope,"JODRELL")==0)
    telescope_id=5;
  else if (strcasecmp(telescope,"GBT")==0)
    telescope_id=6;
  else if (strcasecmp(telescope,"EFFELSBERG")==0)
    telescope_id=8;
  else {
    fprintf(stderr, "Supported telescopes are PARKES, ARECIBO, JODRELL, GBT, EFFELSBERG - setting id to -1\n");
    telescope_id = -1;
   }		

  /* broadcast the header parameters to the output stream */
    send_string("HEADER_START",output);
    send_string("rawdatafile",output);
    send_string(rawfilename,output);
    if (!strings_equal(source_name,"")) {
      send_string("source_name",output);
      send_string(source_name,output);
    }
    printf("sending machine_id..\n");
    send_int("machine_id",machine_id,output);
    send_int("telescope_id",telescope_id,output);
    send_coords(src_raj,src_dej,az_start,za_start,output);
      /* filterbank data */
      /* N.B. for dedisperse to work, foff<0 so flip if necessary */
    send_int("data_type",1,output);
    send_double("fch1",fch1,output);
    send_double("foff",foff,output);
    send_int("nchans",nchans,output);
    /* beam info */
    send_int("nbeams",nbeams,output);
    send_int("ibeam",ibeam,output);
    /* number of bits per sample */
    send_int("nbits",nbits,output);
    /* start time and sample interval */
    send_double("tstart",tstart,output);
    send_double("tsamp",tsamp,output);
    if (nifs == 1) {
      send_int("nifs",1, output);
    } else {
      fprintf(stderr, "Sorry, this program only supports nifs == 1\n");
      exit(1);
      //j=0;
      //for (i=1;i<=nifs;i++) if (ifstream[i-1]=='Y') j++;
      //if (j==0) error_message("no valid IF streams selected!");
      //send_int("nifs",j);
    }
    send_string("HEADER_END",output);		


}	
