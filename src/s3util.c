

/** **************************************************************************
 * s3.c
 *
 * Copyright 2008 Bryan Ischo <bryan@ischo.com>
 *
 * This file is part of libs3.
 *
 * libs3 is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, version 3 of the License.
 *
 * In addition, as a special exception, the copyright holders give
 * permission to link the code of this library and its programs with the
 * OpenSSL library, and distribute linked combinations including the two.
 *
 * libs3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * version 3 along with libs3, in a file named COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 ************************************************************************** **/

#define _XOPEN_SOURCE 600
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


int forceG = 0;
int showResponsePropertiesG = 0;

int retriesG = 5;
int timeoutMsG = 0;
int verifyPeerG = 0;


const char *accessKeyIdG;
const char *secretAccessKeyG;

const char *locationConstraint = 0;
S3CannedAcl cannedAcl = S3CannedAclPrivate;

S3Protocol protocolG = S3ProtocolHTTP;
S3UriStyle uriStyleG = S3UriStylePath;
const char *awsRegionG = NULL;

// Request results, saved as globals -----------------------------------------

int statusG = 0;
char errorDetailsG[4096] = { 0 };


// Other globals -------------------------------------------------------------

char putenvBufG[256];



// util ----------------------------------------------------------------------

void S3_init()
{
    S3Status status;
    const char *hostname = getenv("S3_HOSTNAME");
    
    accessKeyIdG = getenv("S3_ACCESS_KEY_ID");
    if (!accessKeyIdG) {
        fprintf(stderr, "Missing environment variable: S3_ACCESS_KEY_ID\n");
    }
    secretAccessKeyG = getenv("S3_SECRET_ACCESS_KEY");
    if (!secretAccessKeyG) {
        fprintf(stderr,
                "Missing environment variable: S3_SECRET_ACCESS_KEY\n");
    }

    if ((status = S3_initialize("s3", verifyPeerG|S3_INIT_ALL, hostname))
        != S3StatusOK) {
        fprintf(stderr, "Failed to initialize libs3: %s\n",
                S3_get_status_name(status));
        exit(-1);
    }
}


uint64_t convertInt(const char *str, const char *paramName)
{
    uint64_t ret = 0;

    while (*str) {
        if (!isdigit(*str)) {
            fprintf(stderr, "\nERROR: Nondigit in %s parameter: %c\n",
                    paramName, *str);
        }
        ret *= 10;
        ret += (*str++ - '0');
    }

    return ret;
}





// Convenience utility for making the code look nicer.  Tests a string
// against a format; only the characters specified in the format are
// checked (i.e. if the string is longer than the format, the string still
// checks out ok).  Format characters are:
// d - is a digit
// anything else - is that character
// Returns nonzero the string checks out, zero if it does not.
int checkString(const char *str, const char *format)
{
    while (*format) {
        if (*format == 'd') {
            if (!isdigit(*str)) {
                return 0;
            }
        }
        else if (*str != *format) {
            return 0;
        }
        str++, format++;
    }

    return 1;
}


// returns nonzero on success, zero on out of memory
int growbuffer_append(growbuffer **gb, const char *data, int dataLen)
{
    int toCopy = 0 ;
    while (dataLen) {
        growbuffer *buf = *gb ? (*gb)->prev : 0;
        if (!buf || (buf->size == sizeof(buf->data))) {
            buf = (growbuffer *) malloc(sizeof(growbuffer));
            if (!buf) {
                return 0;
            }
            buf->size = 0;
            buf->start = 0;
            if (*gb && (*gb)->prev) {
                buf->prev = (*gb)->prev;
                buf->next = *gb;
                (*gb)->prev->next = buf;
                (*gb)->prev = buf;
            }
            else {
                buf->prev = buf->next = buf;
                *gb = buf;
            }
        }

        toCopy = (sizeof(buf->data) - buf->size);
        if (toCopy > dataLen) {
            toCopy = dataLen;
        }

        memcpy(&(buf->data[buf->size]), data, toCopy);

        buf->size += toCopy, data += toCopy, dataLen -= toCopy;
    }

    return toCopy;
}


void growbuffer_read(growbuffer **gb, int amt, int *amtReturn, char *buffer)
{
    *amtReturn = 0;

    growbuffer *buf = *gb;

    if (!buf) {
        return;
    }

    *amtReturn = (buf->size > amt) ? amt : buf->size;

    memcpy(buffer, &(buf->data[buf->start]), *amtReturn);

    buf->start += *amtReturn, buf->size -= *amtReturn;

    if (buf->size == 0) {
        if (buf->next == buf) {
            *gb = 0;
        }
        else {
            *gb = buf->next;
            buf->prev->next = buf->next;
            buf->next->prev = buf->prev;
        }
        free(buf);
    }
}


void growbuffer_destroy(growbuffer *gb)
{
    growbuffer *start = gb;

    while (gb) {
        growbuffer *next = gb->next;
        free(gb);
        gb = (next == start) ? 0 : next;
    }
}



int should_retry()
{
    if (retriesG--) {
        // Sleep before next retry; start out with a 1 second sleep
        static int retrySleepInterval = 1 * SLEEP_UNITS_PER_SECOND;
        sleep(retrySleepInterval);
        // Next sleep 1 second longer
        retrySleepInterval++;
        return 1;
    }

    return 0;
}




// response properties callback ----------------------------------------------

// This callback does the same thing for every request type: prints out the
// properties if the user has requested them to be so
S3Status responsePropertiesCallback (const S3ResponseProperties *properties, void *callbackData)
{
    (void) callbackData;

    if (!showResponsePropertiesG) {
        return S3StatusOK;
    }

#define print_nonnull(name, field)                                 \
    do {                                                           \
        if (properties-> field) {                                  \
            printf("%s: %s\n", name, properties-> field);          \
        }                                                          \
    } while (0)

    print_nonnull("Content-Type", contentType);
    print_nonnull("Request-Id", requestId);
    print_nonnull("Request-Id-2", requestId2);
    if (properties->contentLength > 0) {
        printf("Content-Length: %llu\n",
               (unsigned long long) properties->contentLength);
    }
    print_nonnull("Server", server);
    print_nonnull("ETag", eTag);
    if (properties->lastModified > 0) {
        char timebuf[256];
        time_t t = (time_t) properties->lastModified;
        // gmtime is not thread-safe but we don't care here.
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
        printf("Last-Modified: %s\n", timebuf);
    }
    int i;
    for (i = 0; i < properties->metaDataCount; i++) {
        printf("x-amz-meta-%s: %s\n", properties->metaData[i].name,
               properties->metaData[i].value);
    }
#ifdef HAS_USESSERVERSIDEENCRYPTION
    if (properties->usesServerSideEncryption) {
        printf("UsesServerSideEncryption: true\n");
    }
#endif

    return S3StatusOK;
}


// response complete callback ------------------------------------------------

// This callback does the same thing for every request type: saves the status
// and error stuff in global variables
void responseCompleteCallback(S3Status status, const S3ErrorDetails *error, void *callbackData)
{
    (void) callbackData;

    statusG = status;
    // Compose the error details message now, although we might not use it.
    // Can't just save a pointer to [error] since it's not guaranteed to last
    // beyond this callback
    int len = 0;
    if (error && error->message) {
        len += snprintf(&(errorDetailsG[len]), sizeof(errorDetailsG) - len,
                        "  Message: %s\n", error->message);
    }
    if (error && error->resource) {
        len += snprintf(&(errorDetailsG[len]), sizeof(errorDetailsG) - len,
                        "  Resource: %s\n", error->resource);
    }
    if (error && error->furtherDetails) {
        len += snprintf(&(errorDetailsG[len]), sizeof(errorDetailsG) - len,
                        "  Further Details: %s\n", error->furtherDetails);
    }
    if (error && error->extraDetailsCount) {
        len += snprintf(&(errorDetailsG[len]), sizeof(errorDetailsG) - len,
                        "%s", "  Extra Details:\n");
        int i;
        for (i = 0; i < error->extraDetailsCount; i++) {
            len += snprintf(&(errorDetailsG[len]),
                            sizeof(errorDetailsG) - len, "    %s: %s\n",
                            error->extraDetails[i].name,
                            error->extraDetails[i].value);
        }
    }
}



// put object ----------------------------------------------------------------



int putObjectDataCallback(int bufferSize, char *buffer, void *callbackData)
{
    put_object_callback_data *data =
        (put_object_callback_data *) callbackData;

    int ret = 0;

    if (data->contentLength) {
        int toRead = ((data->contentLength > (unsigned) bufferSize) ?
                      (unsigned) bufferSize : data->contentLength);
        if (data->gb) {
            growbuffer_read(&(data->gb), toRead, &ret, buffer);
        }
        else if (data->infile) {
            ret = fread(buffer, 1, toRead, data->infile);
        }
    }

    data->contentLength -= ret;
    data->totalContentLength -= ret;

/*
    if (data->contentLength && !data->noStatus) {
        // Avoid a weird bug in MingW, which won't print the second integer
        // value properly when it's in the same call, so print separately
        printf("%llu bytes remaining ",
               (unsigned long long) data->totalContentLength);
        printf("(%d%% complete) ...\n",
               (int) (((data->totalOriginalContentLength -
                        data->totalContentLength) * 100) /
                      data->totalOriginalContentLength));
    }
*/
    return ret;
}








