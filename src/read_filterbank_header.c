/* read_filterbank_header.c - general handling routines for SIGPROC headers */
/* heisted and mangled from sigproc 4.3 Oct 2016 AS */
#include "filterbank_header.h"
#include "filterbankutil.h"



int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) {
    return 1;
  } else {
    return 0;
  }
}

/* read a string from the input which looks like nchars-char[1-nchars] */
void get_string(FILE *inputfile, int *nbytes, char string[])
{
  int nchar;
  strcpy(string,"ERROR");
  fread(&nchar, sizeof(int), 1, inputfile);
  *nbytes=sizeof(int);
  if (feof(inputfile)) exit(0);
  if (nchar>80 || nchar<1) return;
  fread(string, nchar, 1, inputfile);
  string[nchar]='\0';
  *nbytes+=nchar;
}

/* attempt to read in the general header info from a pulsar data file */
int read_filterbank_header(struct filterbank_input *input)
{
  char string[80], message[80];
  int itmp,nbytes,totalbytes,expecting_rawdatafile=0,expecting_source_name=0; 
  int expecting_frequency_table=0,channel_index;
  input->isign=0;

  /* try to read in the first line of the header */
  get_string(input->inputfile,&nbytes,string);
  if (!strings_equal(string,"HEADER_START")) {
	/* the data file is not in standard format, rewind and return */
	rewind(input->inputfile);
	return 0;
  }
  /* store total number of bytes read so far */
  totalbytes=nbytes;

  /* loop over and read remaining header lines until HEADER_END reached */
  while (1) {
    get_string(input->inputfile,&nbytes,string);
    if (strings_equal(string,"HEADER_END")) break;
    totalbytes+=nbytes;
    if (strings_equal(string,"rawdatafile")) {
      expecting_rawdatafile=1;

    } else if (strings_equal(string,"source_name")) {
      expecting_source_name=1;

    } else if (strings_equal(string,"FREQUENCY_START")) {
      expecting_frequency_table=1;
      channel_index=0;

    } else if (strings_equal(string,"FREQUENCY_END")) {
      expecting_frequency_table=0;

    } else if (strings_equal(string,"az_start")) {
      fread(&(input->az_start),sizeof(input->az_start),1,input->inputfile);
      totalbytes+=sizeof(input->az_start);

    } else if (strings_equal(string,"za_start")) {
      fread(&(input->za_start),sizeof(input->za_start),1,input->inputfile);
      totalbytes+=sizeof(input->za_start);

    } else if (strings_equal(string,"src_raj")) {
      fread(&(input->src_raj),sizeof(input->src_raj),1,input->inputfile);
      totalbytes+=sizeof(input->src_raj);

    } else if (strings_equal(string,"src_dej")) {
      fread(&(input->src_dej),sizeof(input->src_dej),1,input->inputfile);
      totalbytes+=sizeof(input->src_dej);

    } else if (strings_equal(string,"tstart")) {
      fread(&(input->tstart),sizeof(input->tstart),1,input->inputfile);
      totalbytes+=sizeof(input->tstart);

    } else if (strings_equal(string,"tsamp")) {
      fread(&(input->tsamp),sizeof(input->tsamp),1,input->inputfile);
      totalbytes+=sizeof(input->tsamp);

    } else if (strings_equal(string,"period")) {
      fread(&(input->period),sizeof(input->period),1,input->inputfile);
      totalbytes+=sizeof(input->period);

    } else if (strings_equal(string,"fch1")) {
      fread(&(input->fch1),sizeof(input->fch1),1,input->inputfile);
      totalbytes+=sizeof(input->fch1);

    } else if (strings_equal(string,"fchannel")) {
      fread(&(input->frequency_table[channel_index++]),sizeof(double),1,input->inputfile);
      totalbytes+=sizeof(double);
      input->fch1=input->foff=0.0; /* set to 0.0 to signify that a table is in use */

    } else if (strings_equal(string,"foff")) {
      fread(&(input->foff),sizeof(input->foff),1,input->inputfile);
      totalbytes+=sizeof(input->foff);

    } else if (strings_equal(string,"nchans")) {
      fread(&(input->nchans),sizeof(input->nchans),1,input->inputfile);
      totalbytes+=sizeof(input->nchans);

    } else if (strings_equal(string,"telescope_id")) {
      fread(&(input->telescope_id),sizeof(input->telescope_id),1,input->inputfile);
      totalbytes+=sizeof(input->telescope_id);

    } else if (strings_equal(string,"machine_id")) {
      fread(&(input->machine_id),sizeof(input->machine_id),1,input->inputfile);
      totalbytes+=sizeof(input->machine_id);

    } else if (strings_equal(string,"data_type")) {
      fread(&(input->data_type),sizeof(input->data_type),1,input->inputfile);
      totalbytes+=sizeof(input->data_type);

    } else if (strings_equal(string,"ibeam")) {
      fread(&(input->ibeam),sizeof(input->ibeam),1,input->inputfile);
      totalbytes+=sizeof(input->ibeam);

    } else if (strings_equal(string,"nbeams")) {
      fread(&(input->nbeams),sizeof(input->nbeams),1,input->inputfile);
      totalbytes+=sizeof(input->nbeams);

    } else if (strings_equal(string,"nbits")) {
      fread(&(input->nbits),sizeof(input->nbits),1,input->inputfile);
      totalbytes+=sizeof(input->nbits);

    } else if (strings_equal(string,"barycentric")) {
      fread(&(input->barycentric),sizeof(input->barycentric),1,input->inputfile);
      totalbytes+=sizeof(input->barycentric);

    } else if (strings_equal(string,"pulsarcentric")) {
      fread(&(input->pulsarcentric),sizeof(input->pulsarcentric),1,input->inputfile);
      totalbytes+=sizeof(input->pulsarcentric);

    } else if (strings_equal(string,"nbins")) {
      fread(&(input->nbins),sizeof(input->nbins),1,input->inputfile);
      totalbytes+=sizeof(input->nbins);

    } else if (strings_equal(string,"nsamples")) {
      /* read this one only for backwards compatibility */
      fread(&itmp,sizeof(itmp),1,input->inputfile);
      totalbytes+=sizeof(itmp);

    } else if (strings_equal(string,"nifs")) {
      fread(&(input->nifs),sizeof(input->nifs),1,input->inputfile);
      totalbytes+=sizeof(input->nifs);

    } else if (strings_equal(string,"npuls")) {
      fread(&(input->npuls),sizeof(input->npuls),1,input->inputfile);
      totalbytes+=sizeof(input->npuls);

    } else if (strings_equal(string,"refdm")) {
      fread(&(input->refdm),sizeof(input->refdm),1,input->inputfile);
      totalbytes+=sizeof(input->refdm);

    } else if (strings_equal(string,"signed")) {
      fread(&(input->isign),sizeof(input->isign),1,input->inputfile);
      totalbytes+=sizeof(input->isign);

    } else if (expecting_rawdatafile) {
      strcpy(input->rawdatafile,string);
      expecting_rawdatafile=0;

    } else if (expecting_source_name) {
      strcpy(input->source_name,string);
      expecting_source_name=0;

    } else {
      sprintf(message,"read_header - unknown parameter: %s\n",string);
      fprintf(stderr,"ERROR: %s\n",message);
      exit(1);
    } 
    if (totalbytes != ftell(input->inputfile)){
	    fprintf(stderr,"ERROR: Header bytes does not equal file position\n");
	    fprintf(stderr,"String was: '%s'\n",string);
	    fprintf(stderr,"       header: %d file: %ld\n",totalbytes,ftell(input->inputfile));
	    exit(1);
    }


  } 

  /* add on last header string */
  totalbytes+=nbytes;

  if (totalbytes != ftell(input->inputfile)){
	  fprintf(stderr,"ERROR: Header bytes does not equal file position\n");
	  fprintf(stderr,"       header: %d file: %ld\n",totalbytes,ftell(input->inputfile));
	  exit(1);
  }

  input->datasize = sizeof_file(input->filename);

  input->headersize = (long int) totalbytes;
   
  input->datasize = input->datasize - (long int) input->headersize;
  
  input->nsamples=(long long) (long double) input->datasize/ (((long double) input->nbits) / 8.0) 
		 /(long double) input->nifs/(long double) input->nchans;
  fprintf(stderr,"header: %ld datasize  %ld nchans %ld \n",input->headersize, input->datasize, input->nchans );




  /* return total number of bytes read */
  return totalbytes;
}
