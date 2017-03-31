#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include "libs3.h"
#include "s3util.h"



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


int main(int argc, char **argv)
{
  

// create bucket -------------------------------------------------------------

	char bucketName[255];
	sprintf(bucketName, "testbucket");


	S3_init();

    S3ResponseHandler responseHandler =
    {
        &responsePropertiesCallback, &responseCompleteCallback
    };

    printf("attempting bucket create...%s %s %d\n", secretAccessKeyG, accessKeyIdG, retriesG);
    do {
        S3_create_bucket(protocolG, accessKeyIdG, secretAccessKeyG, 0, 0,
                         bucketName, awsRegionG, cannedAcl, locationConstraint,
                         0, 0, &responseHandler, 0);
        printf("%s\n", S3_get_status_name(statusG));
    } while (S3_status_is_retryable(statusG) && should_retry());

    if (statusG == S3StatusOK) {
        printf("Bucket successfully created.\n");
    }
    else {
        printf("ERROR.\n");
    }

    
    char key[255];
	char object[1024];
	sprintf(key, "testobject");
	sprintf(object, "THIS IS A TEST OBJECT\n");
    uint64_t contentLength = 1024;


    const char *uploadId = 0;
    const char *cacheControl = 0, *contentType = 0, *md5 = 0;
    const char *contentDispositionFilename = 0, *contentEncoding = 0;
    int64_t expires = -1;
    int metaPropertiesCount = 0;
    S3NameValue metaProperties[S3_MAX_METADATA_COUNT];
    char useServerSideEncryption = 0;
    int noStatus = 0;


    put_object_callback_data data;

    data.infile = 0;
    data.gb = 0;
    data.noStatus = noStatus;

     if (!growbuffer_append(&(data.gb), object, contentLength)) {
         fprintf(stderr, "\nERROR: Out of memory while reading "
                 "stdin\n");
         exit(-1);
     }

    data.totalContentLength =
    data.totalOriginalContentLength =
    data.contentLength =
    data.originalContentLength =
            contentLength;
            
            

    S3BucketContext bucketContext =
    {
        0,
        bucketName,
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


        do {
            S3_put_object(&bucketContext, key, contentLength, &putProperties, 0,
                          0, &putObjectHandler, &data);
        } while (S3_status_is_retryable(statusG) && should_retry());


        if (statusG != S3StatusOK) {
        	printf("ERROR.\n");
        }

        growbuffer_destroy(data.gb);

		sprintf(key, "testobject2");
		sprintf(object, "THIS IS ANOTHER TEST OBJECT\n");

		data.infile = 0;
		data.gb = 0;
		data.noStatus = noStatus;

		if (!growbuffer_append(&(data.gb), object, contentLength)) {
			fprintf(stderr, "\nERROR: Out of memory while reading "
					"stdin\n");
			exit(-1);
		}

		data.totalContentLength =
		data.totalOriginalContentLength =
		data.contentLength =
		data.originalContentLength =
				contentLength;

      do {
            S3_put_object(&bucketContext, key, contentLength, &putProperties, 0,
                          0, &putObjectHandler, &data);
        } while (S3_status_is_retryable(statusG) && should_retry());


        if (statusG != S3StatusOK) {
        	printf("ERROR.\n");
        }



    S3_deinitialize();


    return 0;
}