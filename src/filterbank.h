/* filterbank.h - include file for filterbank and related routines - hacked out from DSPSR stuff Aug/2011 - AS */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* input and output files and logfile (filterbank.monitor) */
FILE *input, *output, *logfile;
char  inpfile[80], outfile[80];

void send_coords(double raj, double dej, double az, double za) ;
void send_double (char *name, double double_precision) ;
void send_int(char *name, int integer) ;
void send_string(char *string) ;
int strings_equal (char *string1, char *string2) ;
void error_message(char *message) ;
void filterbank_header(FILE *outptr) ;


void swap_double( double *pd ) ;
void swap_float( float *pf ) ;
void swap_int( int *pi ) ;
void swap_long( long *pi ) ;

/* global variables describing the data */
#include "header.h"
double time_offset;

/* global variables describing the operating mode */
float start_time, final_time, clip_threshold;
char sigproc_verbose;

int obits, sumifs, headerless, headerfile, swapout, invert_band;
int compute_spectra, do_vanvleck, hanning, hamming, zerolagdump;
int headeronly;
char ifstream[8];
int swapout;

double src_raj;        /* Source RA  (J2000) in hhmmss.ss */
double src_dej;        /* Source DEC (J2000) in ddmmss.ss */
double az_start;       /* Starting azimuth in deg */
double za_start;       /* Starting zenith angle in deg */
