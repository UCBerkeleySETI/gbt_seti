#include "filterbankutil.h"

void filterbank2fits(char * fitsdata, float *datavec, int nchan, int nsamp, long int hitchan, double snr, double doppler, struct filterbank_input *input)
{

char * buf;
long int i,j,k;

buf = (char *) malloc(2880 * sizeof(char));
/* zero out header */
memset(buf, 0x0, 2880);

	strcpy (buf, "END ");
	for(i=4;i<2880;i++) buf[i] = ' ';

	hlength (buf, 2880);

	hadd(buf, "SOURCE");
	hadd(buf, "SNR");
	hadd(buf, "DOPPLER");
	hadd(buf, "RA");
	hadd(buf, "DEC");
	hadd(buf, "MJD");
	hadd(buf, "FCNTR");
	hadd(buf, "DELTAF");
	hadd(buf, "DELTAT");
	hadd(buf, "NAXIS2");
	hadd(buf, "NAXIS1");					 
	hadd(buf, "NAXIS");
	hadd(buf, "BITPIX");
	hadd(buf, "SIMPLE");


	hputc(buf, "SIMPLE", "T");
	hputi4(buf, "BITPIX", -32);
	hputi4(buf, "NAXIS", 2);
	hputi4(buf, "NAXIS1", nchan);
	hputi4(buf, "NAXIS2", nsamp);
	hputnr8(buf, "FCNTR", 12, filterbank_chan_freq(input, hitchan) );
	hputnr8(buf, "DELTAF", 12, (double) input->foff);
	hputnr8(buf, "DELTAT", 12, (double) input->tsamp);

	hputnr8(buf, "MJD", 12, input->tstart);
	hputnr8(buf, "RA", 12, input->src_raj);
	hputnr8(buf, "DEC", 12, input->src_dej);
	hputnr8(buf, "DOPPLER", 12, doppler);
	hputnr8(buf, "SNR", 12, snr);
	hputc(buf, "SOURCE", input->source_name);

	memcpy(fitsdata, buf, 2880 * sizeof(char));
	
	imswap4((char *) datavec,(nchan * nsamp) * 4);
	
	memcpy(fitsdata+2880, datavec, (nchan * nsamp) * 4);
	
	/* create zero pad buffer */
	memset(buf, 0x0, 2880);
	for(i=0;i<2880;i++) buf[i] = ' ';
	
	memcpy(fitsdata + 2880 + (nchan * nsamp * 4), buf, 2880 - ((nchan * nsamp *4)%2880));
	free(buf);
}

double filterbank_chan_freq(struct filterbank_input *input, long int channel) {
	
	return (double) input->fch1 + ((double) channel) * input->foff;

}


void comp_stats(double *mean, double *stddev, float *vec, long int veclen){

	//compute mean and stddev of floating point vector vec, ignoring elements in ignore != 0
	long int i,j,k;
	double tmean = 0;
	double tstddev = 0;
	long int valid_points=0;
		
	for(i=0;i<veclen;i++) {
			tmean = tmean + (double) vec[i];
			tstddev = tstddev + ((double) vec[i] * vec[i]);
			valid_points++;
	}
	
	tstddev = pow((tstddev - ((tmean * tmean)/valid_points))/(valid_points - 1), 0.5);
	tmean = tmean / (valid_points);	
	
	*mean = tmean;
	*stddev = tstddev;

}

void normalize (float *vec, long int veclen) {

	double tmpmean;
	double tmpstd;
	
	float tmpmeanf;
	float tmpstdf;
	long int i;
	comp_stats(&tmpmean, &tmpstd, vec, veclen);

	tmpmeanf = (float) tmpmean;
	tmpstdf = (float) tmpstd;

	/* normalize */
    for(i=0;i<veclen;i++) vec[i] = (vec[i] - tmpmeanf)/ tmpstdf;

}


long int sizeof_file(char name[]) /* includefile */
{
     struct stat stbuf;

     if(stat(name,&stbuf) == -1)
     {
          fprintf(stderr, "f_siz: can't access %s\n",name);
          exit(0);
     }

     return stbuf.st_size;
}

long int filterbank_extract_from_file(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input) {
	long int i,j;
	rewind(input->inputfile);
	fseek(input->inputfile, input->headersize, SEEK_CUR);
	fseek(input->inputfile, tstart * input->nchans * sizeof(float), SEEK_CUR);
	fseek(input->inputfile, chanstart * sizeof(float), SEEK_CUR);
	
	i=0;
	j=0;
	
	while (i < (tend-tstart) ) {	
		 fread(output + (chanend - chanstart) * i, sizeof(char), (chanend - chanstart) * sizeof(float), input->inputfile);  
		 fseek(input->inputfile, (input->nchans - (chanend-chanstart)) * sizeof(float), SEEK_CUR);
		 i++;
	}
	return i;
}


long int candsearch(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char fitsname[100];
	float *snap;
	long int startchan;
	long int endchan;
	
	int goodcandidate = 0;
	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	fitsdata = (char *) malloc(fitslen);
  	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));

	for(i=0;i<input->nchans;i++) {
	
		if (diff_spectrum[i] > thresh) {
		    goodcandidate = 1;
			startchan = i - candwidth/2;
			endchan = i + candwidth/2;

			
			if(endchan > input->nchans) {

				for (j = startchan; j < input->nchans; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 fprintf(stderr, "A %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, input->nchans, input));
				}

			} else if(startchan < 0)    {
				
				for (j = 0; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 					
					
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 fprintf(stderr, "B %ld \n", filterbank_extract_from_file(snap+labs(startchan), 0, input->nsamples, 0, endchan, input));
				}
	
			} else {
				for (j = startchan; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 
				if(goodcandidate == 1) {
					fprintf(stderr, "C %ld %ld %ld %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input), startchan, endchan, input->nsamples);
				}
			}
			
			if(goodcandidate == 1) {

				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snap, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, input);
				   sprintf(fitsname, "./%s_%5.5f_%ld.fits", input->source_name, input->tstart, i);
				   fitsfile = fopen(fitsname, "wb");
				   fwrite(fitsdata, 1, fitslen, fitsfile);
				   fclose(fitsfile);
			}
		}
	
	}
	free(fitsdata);
	free(snap);
	
}	


long int candsearch_onoff(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input, struct filterbank_input *offsource) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char fitsname[100];
	float *snap;
	float *snapoff;
	
	long int startchan;
	long int endchan;
	
	int goodcandidate = 0;
	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	fitsdata = (char *) malloc(fitslen);
  	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));
  	snapoff = (float*) malloc(candwidth * offsource->nsamples * sizeof(float));

	for(i=0;i<input->nchans;i++) {
	
		if (diff_spectrum[i] > thresh) {
		    goodcandidate = 1;
			startchan = i - candwidth/2;
			endchan = i + candwidth/2;

			
			if(endchan > input->nchans) {

				for (j = startchan; j < input->nchans; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 memset(snapoff, 0x0, candwidth * input->nsamples * sizeof(float));
					 fprintf(stderr, "A %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, input->nchans, input));
					 fprintf(stderr, "A %ld \n", filterbank_extract_from_file(snapoff, 0, input->nsamples, startchan, input->nchans, offsource));

				}

			} else if(startchan < 0)    {
				
				for (j = 0; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 					
					
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 memset(snapoff, 0x0, candwidth * input->nsamples * sizeof(float));

					 fprintf(stderr, "B %ld \n", filterbank_extract_from_file(snap+labs(startchan), 0, input->nsamples, 0, endchan, input));
					 fprintf(stderr, "B %ld \n", filterbank_extract_from_file(snapoff+labs(startchan), 0, input->nsamples, 0, endchan, offsource));

				}
	
			} else {
				for (j = startchan; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 
				if(goodcandidate == 1) {
					fprintf(stderr, "C %ld %ld %ld %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input), startchan, endchan, input->nsamples);
					fprintf(stderr, "C %ld %ld %ld %ld \n", filterbank_extract_from_file(snapoff, 0, input->nsamples, startchan, endchan, offsource), startchan, endchan, input->nsamples);

				}
			}
			
			if(goodcandidate == 1) {
				   //for(j=0;j<(candwidth*input->nsamples);j++) snap[j] = (snap[j] - snapoff[j])/snapoff[j];

				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snap, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, input);
				   sprintf(fitsname, "./%s_%5.5f_%ld.fits", input->source_name, input->tstart, i);
				   fitsfile = fopen(fitsname, "wb");
				   fwrite(fitsdata, 1, fitslen, fitsfile);
				   fclose(fitsfile);

				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snapoff, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, input);
				   sprintf(fitsname, "./%s_%5.5f_%ld_OFF.fits", input->source_name, input->tstart, i);
				   fitsfile = fopen(fitsname, "wb");
				   fwrite(fitsdata, 1, fitslen, fitsfile);
				   fclose(fitsfile);



			}
		}
	
	}
	free(fitsdata);
	free(snap);
	free(snapoff);
}	




int sum_filterbank(struct filterbank_input *input) {
	long int i,j,k;
    input->integrated_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->integrated_spectrum, 0x0, input->nchans * sizeof(float));

    input->temp_spectrum = (float*) malloc(input->nchans * sizeof(float));
	memset(input->temp_spectrum, 0x0, input->nchans * sizeof(float));
	j=0;
    while (fread(input->temp_spectrum, sizeof(float), input->nchans, input->inputfile) ) {
           for(i=0;i<input->nchans;i++) input->integrated_spectrum[i] =  input->integrated_spectrum[i] + input->temp_spectrum[i];
    	   j++;
    }
    return j;
}



