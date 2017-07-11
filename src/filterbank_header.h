
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "imswap.h"
#include <sys/stat.h>
#include "mysql.h"


#pragma once
 
/* filterbank input structure */

struct filterbank_input {
	char *filename;
	FILE *inputfile;
	
	char rawdatafile[80], source_name[80];
	int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number,
	  barycentric,pulsarcentric; /* these two added Aug 20, 2004 DRL */
	double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
	double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;
	int nbeams, ibeam;
	char isign;
    
	/* added 20 December 2000    JMC */
	double srcl,srcb;
	double ast0, lst0;
	long wapp_scan_number;
	char project[8];
	char culprits[24];
	double analog_power[2];
	float *integrated_spectrum;
	float *temp_spectrum;
	/* added frequency table for use with non-contiguous data */
	double frequency_table[4096]; /* note limited number of channels */
	long int npuls; /* added for binary pulse profile format */
    long int nsamples;
    long int datasize;
    int headersize;
    int nbins;
	double period;
	long int polychannels;  //Number of polyphase channels in the file
	long int candwidth;
	long int zapwidth;
	long int currentstartchan;


	float *rawdata;
	float *data;
    float *datarev;
    float *result;
    float *revresult;

	float *maxsnr;
	float *maxdrift;
	float *maxsnrrev;
	float *maxdriftrev;

	long int dimY;
	long int dimX;
	long int Xpadframes;

	FILE *candfile;
	char *bucketname;
	char *folder;
	char *diskfolder;
 	char *obsid;
	MYSQL *conn;
};





int read_filterbank_header(struct filterbank_input *input);
