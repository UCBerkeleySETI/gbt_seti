#include "filterbankutil.h"
#include "libs3.h"
#include "s3util.h"
#include <mysql.h>
#include "setimysql.h"
#include <bson.h>
#include <bcon.h>
#include <mongoc.h>
#include "median.h"

#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

// main ----------------------------------------------------------------------
extern int forceG;
extern int showResponsePropertiesG;

extern int retriesG;
extern int timeoutMsG;
extern int verifyPeerG;

extern S3Protocol protocolG;
extern S3UriStyle uriStyleG;
extern char *awsRegionG;

// Environment variables, saved as globals ----------------------------------
extern char *accessKeyIdG;
extern char *secretAccessKeyG;
// Request results, saved as globals -----------------------------------------
extern int statusG;
extern char errorDetailsG[4096];

extern char *locationConstraint;
extern S3CannedAcl cannedAcl;

// Other globals -------------------------------------------------------------
extern char putenvBufG[256];
     
void filterbank2fits(char * fitsdata, float *datavec, int nchan, int nsamp, long int hitchan, double snr, double doppler, struct filterbank_input *input)
{

char * buf;
long int i,j,k;

	printf("in filterbank2fits\n");
	fflush(stdout);
	
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
	
	//for(i=1000;i<1010;i++) printf("in comp stats %g \n", vec[i]);
	
	tstddev = pow((tstddev - ((tmean * tmean)/valid_points))/(valid_points - 1), 0.5);
	tmean = tmean / (valid_points);	
	
	*mean = tmean;
	*stddev = tstddev;

}

void comp_stats_mad(double *median, double *mad, float *vec, long int veclen){

	//compute mean and MAD of floating point vector vec
	long int i,j,k;
	double tmean = 0;
	double tstddev = 0;
	long int valid_points=0;
	float fmedian;
	float fmad;
	float *tempvec;

	tempvec = (float *) malloc(veclen * sizeof(float));

	memcpy(tempvec, vec, veclen * sizeof(float));

	fmedian = median(tempvec, veclen);
	
	for (i = 0; i < veclen; i++) {
			tempvec[i] =  fabsf(vec[i] - fmedian);	 	  		
	}

	fmad = median(tempvec, veclen);		
	free(tempvec);
	
	*median = (double) fmedian;
	*mad = (double) fmad;

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

	for(i=1000;i<1002;i++) printf("in normalize MEAN: %g %g %g \n", vec[i], tmpmeanf, tmpstdf);

	comp_stats_mad(&tmpmean, &tmpstd, vec, veclen);


	tmpmeanf = (float) tmpmean;
	tmpstdf = (float) tmpstd;

	for(i=1000;i<1002;i++) printf("in normalize MEDIAN %g %g %g \n", vec[i], tmpmeanf, tmpstdf);

	//for(i=1000;i<1010;i++) printf("innormalize %g %g %g \n", vec[i], tmpmeanf, tmpstdf);

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


long int filterbank_extract_from_buffer(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input) {
	long int i,j, k, m, n, left, right;
	i=0;
	j=0;
	long int nchanraw;
	nchanraw = input->dimX - (input->Xpadframes * 2 * input->dimY);
	
	if(chanend > nchanraw) {

		 for(i = tstart;i<tend;i++) {
			   memcpy(output + (chanend - chanstart) * i, input->rawdata + (nchanraw * i), (nchanraw - chanstart) * sizeof(float));
		 }
	
	} else if (chanstart < 0) {
		 
		 for(i = tstart;i<tend;i++) {
			   memcpy(output + labs(chanstart) + ((chanend + labs(chanstart)) * i), input->rawdata + (nchanraw * i), chanend * sizeof(float));
		 }
			
	} else { 

		 for(i = tstart;i<tend;i++) {
			   memcpy(output + (chanend - chanstart) * i, input->rawdata + (nchanraw * i) + chanstart, (chanend - chanstart) * sizeof(float));
		 }

	}
	
	return i;
}



long int filterbank_extract_from_file(float *output, long int tstart, long int tend, long int chanstart, long int chanend, struct filterbank_input *input) {
	long int i,j, k, m, n, left, right;
	float mean=0;
	rewind(input->inputfile);
	fseek(input->inputfile, input->headersize, SEEK_CUR);
	fseek(input->inputfile, tstart * input->nchans * sizeof(float), SEEK_CUR);
	
	i=0;
	j=0;
	
	
	if(chanend > input->nchans) {
		fseek(input->inputfile, chanstart * sizeof(float), SEEK_CUR);

		  while (i < (tend-tstart) ) {	
			   fread(output + (chanend - chanstart) * i, sizeof(char), (input->nchans - (chanstart)) * sizeof(float), input->inputfile);  
			   fseek(input->inputfile, (chanstart) * sizeof(float), SEEK_CUR);
			   i++;
		  }		
	
	} else if (chanstart < 0) {
		  while (i < (tend-tstart) ) {	
			   fread((output + labs(chanstart)) + (chanend + labs(chanstart)) * i, sizeof(char), (chanend) * sizeof(float), input->inputfile);  
			   fseek(input->inputfile, (input->nchans - (chanend)) * sizeof(float), SEEK_CUR);
			   i++;
		  }	
	} else { 
		fseek(input->inputfile, chanstart * sizeof(float), SEEK_CUR);

		  while (i < (tend-tstart) ) {	
			   fread(output + (chanend - chanstart) * i, sizeof(char), (chanend - chanstart) * sizeof(float), input->inputfile);  
			   fseek(input->inputfile, (input->nchans - (chanend-chanstart)) * sizeof(float), SEEK_CUR);
			   i++;
		  }
	}
	
	//loop from chanstart to chanend

	for(j = chanstart; j < chanend; j++) {
	    if( (j + (input->nchans/input->polychannels)/2)%(input->nchans/input->polychannels) == 0 ) {

			  i = j - chanstart;
			    			  
			  left =  i - (long int) ceil((input->zapwidth + 1) / 2);
			  right = i + (long int) floor((input->zapwidth + 1) / 2);
			  //fprintf(stderr, "left: %ld, right %ld\n", left, right);
			  
			  
			  for(m = 0; m < (tend-tstart); m++) {

				   if(left >= 0 && right < input->nchans) {
						 mean = (output[left+(m * (chanend-chanstart))] + output[right+(m * (chanend-chanstart))])/2;
				   } else if (left < 0 && right < input->nchans) {
						 mean = (output[right+(m * (chanend-chanstart))]);
				   } else if (left >= 0 && right >= input->nchans) {
						 mean = (output[left+(m * (chanend-chanstart))]);
				   }



				   for(k = max(0, (left+1));k < min(right, input->nchans);k++) {
						output[k + (m * (chanend-chanstart))] = mean;				   
						}			  
			  }

		  }					  
			  
	    }

	
	//if (channel + 1/2 polyphase length) % polyphase length == 0
	
	
            //loop over all integrations, setting DC channel + buffer
	
	
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
					 fprintf(stderr, "A %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input));
				}

			} else if(startchan < 0)    {
				
				for (j = 0; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 					
					
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 fprintf(stderr, "B %ld \n", filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input), startchan, endchan, input->nsamples);
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






long int candsearch_doppler(float thresh, struct filterbank_input *input, struct filterbank_input *offsource) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char fitsname[100];
	char candname[150];
	float *snap;
	float *snapoff;
	
	long int startchan;
	long int endchan;
	long int onout;
	long int offout;
	
	
	int goodcandidate = 0;
	char query[4096];
	MYSQL *conn;

	//char bucketName[255];
	//sprintf(bucketName, "norwegianfits");


	S3_init();
	dbconnect(&conn);

	long int candwidth;
	
	candwidth = input->candwidth;

    const char *uploadId = 0;
    const char *cacheControl = 0, *contentType = 0, *md5 = 0;
    const char *contentDispositionFilename = 0, *contentEncoding = 0;
    int64_t expires = -1;
    int metaPropertiesCount = 0;
    S3NameValue metaProperties[S3_MAX_METADATA_COUNT];
    char useServerSideEncryption = 0;
    int noStatus = 0;


    put_object_callback_data data;

	S3BucketContext bucketContext =
    {
        0,
        input->bucketname,
        protocolG,
        uriStyleG,
        accessKeyIdG,
        secretAccessKeyG,
        0,
        awsRegionG
    };

    S3PutProperties putProperties =
    {
        contentType,
        md5,
        cacheControl,
        contentDispositionFilename,
        contentEncoding,
        expires,
        cannedAcl,
        metaPropertiesCount,
        metaProperties,
        useServerSideEncryption
    };
    
    S3PutObjectHandler putObjectHandler =
    {
        { &responsePropertiesCallback, &responseCompleteCallback },
        &putObjectDataCallback
    };

	
	
	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	fitsdata = (char *) malloc(fitslen);
	
	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));
	snapoff = (float*) malloc(candwidth * input->nsamples * sizeof(float));
	
		
	float drift_rate_resolution = (input->foff * input->nsamples * input->tsamp);
	float onsourcesnr = 0;
	float offsourcesnr = 0;
	double drift = 0;
	double frequency = 0;
	long int finechannel = 0;

	
	for(j=(input->Xpadframes * input->dimY);j<(input->dimX - (input->Xpadframes * input->dimY));j++) {


			goodcandidate = 0;

			if (input->maxsnr[j] > thresh) {

				//printf("%s: Cand at frequency: %15.6lf bandwidth: %15.6g %ld of %ld: onsource snr: %g drift: %g Hz/sec off source snr: %g\n", sourcea.filename, filterbank_chan_freq(&sourcea, j-(sourcea.Xpadframes * sourcea.dimY)), sourcea.foff, j, sourcea.dimX, sourcea.maxsnr[j], sourcea.maxdrift[j] * drift_rate_resolution, sourceb.maxsnr[j]);

				onsourcesnr = input->maxsnr[j];
				offsourcesnr = offsource->maxsnr[j];
				drift = input->maxdrift[j] * drift_rate_resolution;
				finechannel = (j-(input->Xpadframes * input->dimY));				
				frequency = filterbank_chan_freq(input, input->currentstartchan + finechannel);
				
				goodcandidate = 1;
			}
			
			if (input->maxsnrrev[j] > thresh) {
				//printf("%s: Cand at frequency: %15.6lf bandwidth: %15.6g %ld of %ld: onsource snr: %g drift: %g Hz/sec off source snr: %g\n",sourcea.filename, filterbank_chan_freq(&sourcea, j-(sourcea.Xpadframes * sourcea.dimY)), sourcea.foff, j, sourcea.dimX, sourcea.maxsnrrev[j], sourcea.maxdrift[j] * drift_rate_resolution * -1, sourceb.maxsnrrev[j]);
	
				onsourcesnr = input->maxsnrrev[j];
				offsourcesnr = offsource->maxsnrrev[j];
				drift = input->maxdriftrev[j] * drift_rate_resolution * -1;
				finechannel = (j-(input->Xpadframes * input->dimY));				
				frequency = filterbank_chan_freq(input, input->currentstartchan + finechannel);
				goodcandidate = 1;
			}
		 
		 
		 if(goodcandidate == 1) {
		 	startchan = finechannel - candwidth/2;
			endchan = finechannel + candwidth/2;
		 
		 
				memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
				memset(snapoff, 0x0, candwidth * input->nsamples * sizeof(float));


				onout = filterbank_extract_from_buffer(snap, 0, input->nsamples, startchan, endchan, input);
				offout = filterbank_extract_from_buffer(snapoff, 0, offsource->nsamples, startchan, endchan, offsource);
					
					
				sprintf(query, "INSERT INTO hits (source, mjd, frequency, finechannel, zscore, bw, driftrate) VALUES (\"%s\", %5.5f, %f, %ld, %lf, %6.10f, %lf)", input->source_name, input->tstart, frequency, input->currentstartchan +finechannel, onsourcesnr, fabs(input->foff), drift);



				//printf("%s\n", query);
				
				   
				  if (mysql_query(conn,query)){
					if(mysql_errno(conn) == 1062) {
					fprintf(stderr, "Duplicate! %d\n", mysql_errno(conn));
					printf("INSERT INTO hits (source, mjd, frequency, finechannel, zscore, bw, driftrate) VALUES (\"%s\", %5.5f, %f, %ld, %lf, %6.10f, %lf)", input->source_name, input->tstart, frequency, input->currentstartchan +finechannel, onsourcesnr, fabs(input->foff), drift);

					} else {

					fprintf(stderr, "Error inserting raw hit into sql database...%d\n", mysql_errno(conn));
					exiterr(3);
					}
					
				   }  
				


				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snap, candwidth, input->nsamples, input->currentstartchan + finechannel, onsourcesnr, drift, input);

			  	   if (input->folder != NULL) {
   				   		sprintf(fitsname, "%s/%s_%5.5f_%ld.fits", input->folder,input->source_name, input->tstart, input->currentstartchan +finechannel);
				   } else {
				      	 sprintf(fitsname, "%s_%5.5f_%ld.fits",input->source_name, input->tstart, input->currentstartchan +finechannel);

				   }
				   
				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}

				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;

				   do {
					   S3_put_object(&bucketContext, fitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());


				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s\n", fitsname);
				   }

				   growbuffer_destroy(data.gb);


				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snapoff, candwidth, input->nsamples, input->currentstartchan + finechannel, offsourcesnr, 0.0, offsource);
				  
				   if (input->folder != NULL) {
				   		sprintf(fitsname, "%s/%s_%5.5f_%ld_OFF.fits", input->folder,input->source_name, input->tstart, input->currentstartchan +finechannel);
				   } else {
				   		sprintf(fitsname, "%s_%5.5f_%ld_OFF.fits", input->source_name, input->tstart,input->currentstartchan + finechannel);

				   }
	
	
	


				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}
					
				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;
					
				   do {
					   S3_put_object(&bucketContext, fitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());

				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s %s.\n", fitsname, S3_get_status_name(statusG));
				   }

				   growbuffer_destroy(data.gb);
				   

				}
				
			}
			
		
	
	
	free(fitsdata);
	free(snap);
	free(snapoff);
	S3_deinitialize();
	do_disconnect(&conn);

	
}	






long int candsearch_doppler_mongo(float thresh, struct filterbank_input *input, struct filterbank_input *offsource) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char onfitsname[100];
	char offfitsname[100];
	char diskfitsname[200];
	
	
	char candname[150];
	float *snap;
	float *snapoff;
	
	long int startchan;
	long int endchan;
	long int onout;
	long int offout;
	
	char mongoclientstring[1024];
	int goodcandidate = 0;
	char query[4096];

   mongoc_client_t      *client;
   mongoc_database_t    *database;
   mongoc_collection_t  *collection;
   bson_t               *command,
                        *insert;
   bson_error_t          error;
   char                 *str;


   char *envvar = NULL;

	//char bucketName[255];
	//sprintf(bucketName, "norwegianfits");

    envvar = getenv("MONGO_CLIENT_STRING");
    if (!envvar) {
        fprintf(stderr, "Missing environment variable: MONGO_CLIENT_STRING\n");
        exit(1);
    }
	
	sprintf(mongoclientstring, "%s", envvar);
    
    printf("%s\n",mongoclientstring);

	S3_init();
	
	
	/*
	 * Required to initialize libmongoc's internals
	 */
	mongoc_init ();

	/*
	 * Create a new client instance
	 */
 	client = mongoc_client_new (mongoclientstring);

	/*
	 * Register the application name so we can track it in the profile logs
	 * on the server. This can also be done from the URI (see other examples).
	 */
	mongoc_client_set_appname (client, "rt_obs_gb");

	/*
	 * Get a handle on the database "db_name" and collection "coll_name"
	 */
	database = mongoc_client_get_database (client, "observationTracker");
	collection = mongoc_client_get_collection (client, "observationTracker", "events");
   	
   	char cloudbucket[128];
   	sprintf(cloudbucket,"https://storage.googleapis.com/setidata");

	printf("MongoConnected\n");
	fflush(stdout);

	
   
	long int candwidth;
	candwidth = input->candwidth;
	
    const char *cacheControl = 0, *contentType = 0, *md5 = 0;
    const char *contentDispositionFilename = 0, *contentEncoding = 0;
    int64_t expires = -1;
    int metaPropertiesCount = 0;
    


    
    
    S3NameValue metaProperties[S3_MAX_METADATA_COUNT];
    char useServerSideEncryption = 0;
    int noStatus = 0;




    put_object_callback_data data;

	S3BucketContext bucketContext =
    {
        0,
        input->bucketname,
        protocolG,
        uriStyleG,
        accessKeyIdG,
        secretAccessKeyG,
        0,
        awsRegionG
    };

    S3PutProperties putProperties =
    {
        contentType,
        md5,
        cacheControl,
        contentDispositionFilename,
        contentEncoding,
        expires,
        cannedAcl,
        metaPropertiesCount,
        metaProperties,
        useServerSideEncryption
    };
    
    S3PutObjectHandler putObjectHandler =
    {
        { &responsePropertiesCallback, &responseCompleteCallback },
        &putObjectDataCallback
    };


			
	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	
	fitsdata = (char *) malloc(fitslen);
	
	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));
	snapoff = (float*) malloc(candwidth * input->nsamples * sizeof(float));

			
		
	float drift_rate_resolution = (input->foff * input->nsamples * input->tsamp);
	float onsourcesnr = 0;
	float offsourcesnr = 0;
	double drift = 0;
	double frequency = 0;
	long int finechannel = 0;
			

	
	for(j=(input->Xpadframes * input->dimY);j<(input->dimX - (input->Xpadframes * input->dimY));j++) {


			goodcandidate = 0;

			if (input->maxsnr[j] > thresh) {

				//printf("%s: Cand at frequency: %15.6lf bandwidth: %15.6g %ld of %ld: onsource snr: %g drift: %g Hz/sec off source snr: %g\n", sourcea.filename, filterbank_chan_freq(&sourcea, j-(sourcea.Xpadframes * sourcea.dimY)), sourcea.foff, j, sourcea.dimX, sourcea.maxsnr[j], sourcea.maxdrift[j] * drift_rate_resolution, sourceb.maxsnr[j]);

				onsourcesnr = input->maxsnr[j];
				offsourcesnr = offsource->maxsnr[j];
				drift = input->maxdrift[j] * drift_rate_resolution;
				finechannel = (j-(input->Xpadframes * input->dimY));				
				frequency = filterbank_chan_freq(input, input->currentstartchan + finechannel);
				
				goodcandidate = 1;
			}
			
			if (input->maxsnrrev[j] > thresh) {
				//printf("%s: Cand at frequency: %15.6lf bandwidth: %15.6g %ld of %ld: onsource snr: %g drift: %g Hz/sec off source snr: %g\n",sourcea.filename, filterbank_chan_freq(&sourcea, j-(sourcea.Xpadframes * sourcea.dimY)), sourcea.foff, j, sourcea.dimX, sourcea.maxsnrrev[j], sourcea.maxdrift[j] * drift_rate_resolution * -1, sourceb.maxsnrrev[j]);
	
				onsourcesnr = input->maxsnrrev[j];
				offsourcesnr = offsource->maxsnrrev[j];
				drift = input->maxdriftrev[j] * drift_rate_resolution * -1;
				finechannel = (j-(input->Xpadframes * input->dimY));				
				frequency = filterbank_chan_freq(input, input->currentstartchan + finechannel);
				goodcandidate = 1;
			}
		 
		 
		 if(goodcandidate == 1) {
		 	startchan = finechannel - candwidth/2;
			endchan = finechannel + candwidth/2;
		 
		 
				memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
				memset(snapoff, 0x0, candwidth * input->nsamples * sizeof(float));


				onout = filterbank_extract_from_buffer(snap, 0, input->nsamples, startchan, endchan, input);
				offout = filterbank_extract_from_buffer(snapoff, 0, offsource->nsamples, startchan, endchan, offsource);
					
				
			
			  sprintf(query, 
				  "{\"observationId\":\"%s\","  	
				  "\"snr\":%f," 
				  "\"frequency\":%f," 
				  "\"ra\":\"%09.2lf\"," 
				  "\"dec\":\"%09.2lf\"," 
				  "\"bandwidth\":%lf," 
				  "\"starName\":\"%s\"," 
				  "\"driftRate\":%f," 
				  "\"onImgURI\":\"%s/%s_%5.5f_%ld_ON.png\"," 
				  "\"offImgURI\":\"%s/%s_%5.5f_%ld_OFF.png\"," 
				  "\"diffImgURI\":\"%s/%s_%5.5f_%ld_NORM.png\"," 
				  "\"onFitsURI\":\"%s/%s_%5.5f_%ld.fits\"," 
				  "\"offFitsURI\":\"%s/%s_%5.5f_%ld_OFF.fits\"}",
				   input->obsid,
				   onsourcesnr,
				   frequency,
				   input->src_raj + 0.1, //The '+ 0.1' has been added here to mitigate a fatal error
				   input->src_dej + 0.1, //in the obs-tracker package - A.P.V.S. July 21, 2017
				   input->foff*1000000*-1,
				   input->source_name,
				   drift,
				   cloudbucket, input->source_name, input->tstart, input->currentstartchan +finechannel,
				   cloudbucket, input->source_name, input->tstart, input->currentstartchan +finechannel,
				   cloudbucket, input->source_name, input->tstart, input->currentstartchan +finechannel,
				   cloudbucket, input->source_name, input->tstart, input->currentstartchan +finechannel,
				   cloudbucket, input->source_name, input->tstart, input->currentstartchan +finechannel);
	 
	 
	
				printf("%s\n", query);


				//const char *json = "{\"name\": {\"first\":\"Grace\", \"last\":\"Hopper\"}}";
				
				insert = bson_new_from_json ((const uint8_t *)query, -1, &error);
   

				//printf("%s\n", query);
				
				   
				   if (!mongoc_collection_insert (collection, MONGOC_INSERT_NONE, insert, NULL, &error)) {
					  fprintf (stderr, "%s\n", error.message);
				   }


				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snap, candwidth, input->nsamples, input->currentstartchan + finechannel, onsourcesnr, drift, input);

			  	   if (input->folder != NULL) {
   				   		sprintf(onfitsname, "%s/%s_%5.5f_%ld.fits", input->folder,input->source_name, input->tstart, input->currentstartchan +finechannel);
				   } else {
				      	 sprintf(onfitsname, "%s_%5.5f_%ld.fits",input->source_name, input->tstart, input->currentstartchan +finechannel);
				   }
				  
 				   if (input->diskfolder != NULL) {
 
					   sprintf(diskfitsname, "%s/%s", input->diskfolder, onfitsname);
					   fitsfile = fopen(diskfitsname, "wb");
					   fwrite(fitsdata, 1, fitslen, fitsfile);
					   fclose(fitsfile);
				  }
				   
				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}

				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;

				   do {
					   S3_put_object(&bucketContext, onfitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());


				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s\n", onfitsname);
				   }

				   growbuffer_destroy(data.gb);


				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snapoff, candwidth, input->nsamples, input->currentstartchan + finechannel, offsourcesnr, 0.0, offsource);
				  
				   if (input->folder != NULL) {
				   		sprintf(offfitsname, "%s/%s_%5.5f_%ld_OFF.fits", input->folder,input->source_name, input->tstart, input->currentstartchan +finechannel);
				   } else {
				   		sprintf(offfitsname, "%s_%5.5f_%ld_OFF.fits", input->source_name, input->tstart,input->currentstartchan + finechannel);

				   }
	
	
                                   if (input->diskfolder != NULL) {

                                           sprintf(diskfitsname, "%s/%s", input->diskfolder, offfitsname);
                                           fitsfile = fopen(diskfitsname, "wb");
                                           fwrite(fitsdata, 1, fitslen, fitsfile);
                                           fclose(fitsfile);
                                  }	


				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}
					
				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;
					
				   do {
					   S3_put_object(&bucketContext, offfitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());

				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s %s.\n", offfitsname, S3_get_status_name(statusG));
				   }

				   growbuffer_destroy(data.gb);
				   

				}
				
			}
			
		
	
	
	free(fitsdata);
	free(snap);
	free(snapoff);
	S3_deinitialize();

	mongoc_collection_destroy (collection);
	mongoc_database_destroy (database);
	mongoc_client_destroy (client);
	mongoc_cleanup ();
}	













long int candsearch_onoff(float *diff_spectrum, long int candwidth, float thresh, struct filterbank_input *input, struct filterbank_input *offsource) {

	long int i, j, k;
	long int fitslen;
	char *fitsdata;
	FILE *fitsfile;
	char fitsname[100];
	char candname[150];
	float *snap;
	float *snapoff;
	
	long int startchan;
	long int endchan;
	long int onout;
	long int offout;
	
	int goodcandidate = 0;
	char query[4096];
	MYSQL *conn;

	//char bucketName[255];
	//sprintf(bucketName, "norwegianfits");


	S3_init();
	dbconnect(&conn);

	

    const char *uploadId = 0;
    const char *cacheControl = 0, *contentType = 0, *md5 = 0;
    const char *contentDispositionFilename = 0, *contentEncoding = 0;
    int64_t expires = -1;
    int metaPropertiesCount = 0;
    S3NameValue metaProperties[S3_MAX_METADATA_COUNT];
    char useServerSideEncryption = 0;
    int noStatus = 0;


    put_object_callback_data data;

	S3BucketContext bucketContext =
    {
        0,
        input->bucketname,
        protocolG,
        uriStyleG,
        accessKeyIdG,
        secretAccessKeyG,
        0,
        awsRegionG
    };

    S3PutProperties putProperties =
    {
        contentType,
        md5,
        cacheControl,
        contentDispositionFilename,
        contentEncoding,
        expires,
        cannedAcl,
        metaPropertiesCount,
        metaProperties,
        useServerSideEncryption
    };
    
    S3PutObjectHandler putObjectHandler =
    {
        { &responsePropertiesCallback, &responseCompleteCallback },
        &putObjectDataCallback
    };

	
	
	fitslen = 2880 + (candwidth * input->nsamples * 4) + 2880 - ((candwidth * input->nsamples * 4)%2880);
	fitsdata = (char *) malloc(fitslen);
  	snap = (float*) malloc(candwidth * input->nsamples * sizeof(float));
  	snapoff = (float*) malloc(candwidth * offsource->nsamples * sizeof(float));
	
	sprintf(candname, "%s_%f.candidates", input->source_name, input->tstart); 
	input->candfile = fopen(candname, "w");


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
					 onout = filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input);
					 offout = filterbank_extract_from_file(snapoff, 0, input->nsamples, startchan, endchan, offsource);
						
					 //fprintf(stderr, "A %ld %ld %ld %ld \n", onout, startchan, endchan, input->nsamples);
					 //fprintf(stderr, "A %ld %ld %ld %ld \n", offout, startchan, endchan, input->nsamples);


				}

			} else if(startchan < 0)    {
				
				for (j = 0; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 					
					
				if(goodcandidate == 1) {
					 memset(snap, 0x0, candwidth * input->nsamples * sizeof(float));
					 memset(snapoff, 0x0, candwidth * input->nsamples * sizeof(float));
					 onout = filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input);
					 offout = filterbank_extract_from_file(snapoff, 0, input->nsamples, startchan, endchan, offsource);
					
					 //fprintf(stderr, "B %ld %ld %ld %ld \n", onout, startchan, endchan, input->nsamples);
					 //fprintf(stderr, "B %ld %ld %ld %ld \n", offout, startchan, endchan, input->nsamples);

				}
	
			} else {
				for (j = startchan; j < endchan; j++) {
					if (diff_spectrum[j] > diff_spectrum[i]) goodcandidate = 0;
				} 
				if(goodcandidate == 1) {
					onout = filterbank_extract_from_file(snap, 0, input->nsamples, startchan, endchan, input);
					offout = filterbank_extract_from_file(snapoff, 0, input->nsamples, startchan, endchan, offsource);
					
					//fprintf(stderr, "C %ld %ld %ld %ld \n", onout, startchan, endchan, input->nsamples);
					//fprintf(stderr, "C %ld %ld %ld %ld \n", offout, startchan, endchan, input->nsamples);

				}
			}
			
			if(goodcandidate == 1) {
				   //for(j=0;j<(candwidth*input->nsamples);j++) snap[j] = (snap[j] - snapoff[j])/snapoff[j];

//				   fprintf(input->candfile, "INSERT INTO hits (source, %s, %s, %f, %f, %lf, %6.10f \n", input->filename, input->source_name, input->tstart, filterbank_chan_freq(input, i), diff_spectrum[i], fabs(input->foff));
				sprintf(query, "INSERT INTO hits (source, mjd, frequency, finechannel, zscore, bw) VALUES (\"%s\", %5.5f, %f, %ld, %lf, %6.10f)", input->source_name, input->tstart, filterbank_chan_freq(input, i), i, diff_spectrum[i], fabs(input->foff));
				//printf("%s\n", query);
				
				   /*
				sprintf(query, "INSERT INTO rawhits \
				(ra, decl, snr, mjd, topofreq, baryfreq, finebw, src_name, az, za, baryv, barya) \
				VALUES (%15.15f, %15.15f, %15.15f, %15.15Lf, %15.15f, %15.15f, \
				%15.15f, \"%s\", %15.15f, %15.15f, %15.20f, %15.20f)",\				   
				   
				   */
				   
				 if (mysql_query(conn,query)){
					if(mysql_errno(conn) == 1062) {
					fprintf(stderr, "Duplicate! %d\n", mysql_errno(conn));

					} else {

					fprintf(stderr, "Error inserting raw hit into sql database...%d\n", mysql_errno(conn));
					exiterr(3);
					}
					
				}  
				

				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snap, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, input);

				   //sprintf(fitsname, "./%s_%5.5f_%ld.fits", input->source_name, input->tstart, i);
				   //fitsfile = fopen(fitsname, "wb");
				   //fwrite(fitsdata, 1, fitslen, fitsfile);
				   //fclose(fitsfile);
			  
   				   sprintf(fitsname, "%s/%s_%5.5f_%ld.fits", input->folder,input->source_name, input->tstart, i);

				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}

				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;

				   do {
					   S3_put_object(&bucketContext, fitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());


				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s\n", fitsname);
				   }

				   growbuffer_destroy(data.gb);


				   memset(fitsdata, 0x0, fitslen * sizeof(char));
				   filterbank2fits(fitsdata, snapoff, candwidth, input->nsamples, i, diff_spectrum[i], 0.0, offsource);
				   sprintf(fitsname, "%s/%s_%5.5f_%ld_OFF.fits", input->folder,input->source_name, input->tstart, i);

				   data.infile = 0;
				   data.gb = 0;
				   data.noStatus = noStatus;

					if (!growbuffer_append(&(data.gb), fitsdata, fitslen)) {
						fprintf(stderr, "\nERROR: Out of memory while reading "
								"stdin\n");
						exit(-1);
					}
					
				   data.totalContentLength =
				   data.totalOriginalContentLength =
				   data.contentLength =
				   data.originalContentLength =
						   fitslen;
					
				   do {
					   S3_put_object(&bucketContext, fitsname, fitslen, &putProperties, 0,
									 0, &putObjectHandler, &data);
				   } while (S3_status_is_retryable(statusG) && should_retry());

				   if (statusG != S3StatusOK) {
					   printf("ERROR ON PUT OBJECT %s %s.\n", fitsname, S3_get_status_name(statusG));
				   }

				   growbuffer_destroy(data.gb);

				   //sprintf(fitsname, "./%s_%5.5f_%ld_OFF.fits", input->source_name, input->tstart, i);
				   //fitsfile = fopen(fitsname, "wb");
				   //fwrite(fitsdata, 1, fitslen, fitsfile);
				   //fclose(fitsfile);
				

			}
		}
	
	}
	free(fitsdata);
	free(snap);
	free(snapoff);
	fclose(input->candfile);
	S3_deinitialize();
	do_disconnect(&conn);
	
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


void filterbanksearch_print_usage() /*includefile*/
{
  printf("\n");
  printf("filterbanksearch  - search filterbank format data for signals from aliens\n\n");
  printf("usage: filterbanksearch -a <ON source file name> -b <OFF source file name> -{options}\n\n");
  printf("options:\n\n");
  printf("-a <file>      	 	- set ON source file\n");
  printf("-b <file>       		- set OFF source file \n");
  printf("-s <string>     		- set name of S3 bucket\n");
  printf("-f <string>     		- set folder path\n");
  printf("-m <string>       	- set name of MYSQL database\n");
  printf("-t <thresh>         	- set number of bits per sample\n");
  printf("-z <thresh>         	- set threshold for events\n");
  printf("\n");
  printf("Meaningful Environment Variables:\n");  
  printf("\n");
}





