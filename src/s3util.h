#include "libs3.h"

// Some Windows stuff
#ifndef FOPEN_EXTRA_FLAGS
#define FOPEN_EXTRA_FLAGS ""
#endif

// Some Unix stuff (to work around Windows issues)
#ifndef SLEEP_UNITS_PER_SECOND
#define SLEEP_UNITS_PER_SECOND 1
#endif

// Also needed for Windows, because somehow MinGW doesn't define this
extern int putenv(char *);




// Command-line options, saved as globals ------------------------------------


// Option prefixes -----------------------------------------------------------

#define LOCATION_PREFIX "location="
#define LOCATION_PREFIX_LEN (sizeof(LOCATION_PREFIX) - 1)
#define CANNED_ACL_PREFIX "cannedAcl="
#define CANNED_ACL_PREFIX_LEN (sizeof(CANNED_ACL_PREFIX) - 1)
#define PREFIX_PREFIX "prefix="
#define PREFIX_PREFIX_LEN (sizeof(PREFIX_PREFIX) - 1)
#define MARKER_PREFIX "marker="
#define MARKER_PREFIX_LEN (sizeof(MARKER_PREFIX) - 1)
#define DELIMITER_PREFIX "delimiter="
#define DELIMITER_PREFIX_LEN (sizeof(DELIMITER_PREFIX) - 1)
#define ENCODING_TYPE_PREFIX "encoding-type="
#define ENCODING_TYPE_PREFIX_LEN (sizeof(ENCODING_TYPE_PREFIX) - 1)
#define MAX_UPLOADS_PREFIX "max-uploads="
#define MAX_UPLOADS_PREFIX_LEN (sizeof(MAX_UPLOADS_PREFIX) - 1)
#define KEY_MARKER_PREFIX "key-marker="
#define KEY_MARKER_PREFIX_LEN (sizeof(KEY_MARKER_PREFIX) - 1)
#define UPLOAD_ID_PREFIX "upload-id="
#define UPLOAD_ID_PREFIX_LEN (sizeof(UPLOAD_ID_PREFIX) - 1)
#define MAX_PARTS_PREFIX "max-parts="
#define MAX_PARTS_PREFIX_LEN (sizeof(MAX_PARTS_PREFIX) - 1)
#define PART_NUMBER_MARKER_PREFIX "part-number-marker="
#define PART_NUMBER_MARKER_PREFIX_LEN (sizeof(PART_NUMBER_MARKER_PREFIX) - 1)
#define UPLOAD_ID_MARKER_PREFIX "upload-id-marker="
#define UPLOAD_ID_MARKER_PREFIX_LEN (sizeof(UPLOAD_ID_MARKER_PREFIX) - 1)
#define MAXKEYS_PREFIX "maxkeys="
#define MAXKEYS_PREFIX_LEN (sizeof(MAXKEYS_PREFIX) - 1)
#define FILENAME_PREFIX "filename="
#define FILENAME_PREFIX_LEN (sizeof(FILENAME_PREFIX) - 1)
#define CONTENT_LENGTH_PREFIX "contentLength="
#define CONTENT_LENGTH_PREFIX_LEN (sizeof(CONTENT_LENGTH_PREFIX) - 1)
#define CACHE_CONTROL_PREFIX "cacheControl="
#define CACHE_CONTROL_PREFIX_LEN (sizeof(CACHE_CONTROL_PREFIX) - 1)
#define CONTENT_TYPE_PREFIX "contentType="
#define CONTENT_TYPE_PREFIX_LEN (sizeof(CONTENT_TYPE_PREFIX) - 1)
#define MD5_PREFIX "md5="
#define MD5_PREFIX_LEN (sizeof(MD5_PREFIX) - 1)
#define CONTENT_DISPOSITION_FILENAME_PREFIX "contentDispositionFilename="
#define CONTENT_DISPOSITION_FILENAME_PREFIX_LEN \
    (sizeof(CONTENT_DISPOSITION_FILENAME_PREFIX) - 1)
#define CONTENT_ENCODING_PREFIX "contentEncoding="
#define CONTENT_ENCODING_PREFIX_LEN (sizeof(CONTENT_ENCODING_PREFIX) - 1)
#define EXPIRES_PREFIX "expires="
#define EXPIRES_PREFIX_LEN (sizeof(EXPIRES_PREFIX) - 1)
#define X_AMZ_META_PREFIX "x-amz-meta-"
#define X_AMZ_META_PREFIX_LEN (sizeof(X_AMZ_META_PREFIX) - 1)
#define USE_SERVER_SIDE_ENCRYPTION_PREFIX "useServerSideEncryption="
#define USE_SERVER_SIDE_ENCRYPTION_PREFIX_LEN \
    (sizeof(USE_SERVER_SIDE_ENCRYPTION_PREFIX) - 1)
#define IF_MODIFIED_SINCE_PREFIX "ifModifiedSince="
#define IF_MODIFIED_SINCE_PREFIX_LEN (sizeof(IF_MODIFIED_SINCE_PREFIX) - 1)
#define IF_NOT_MODIFIED_SINCE_PREFIX "ifNotmodifiedSince="
#define IF_NOT_MODIFIED_SINCE_PREFIX_LEN \
    (sizeof(IF_NOT_MODIFIED_SINCE_PREFIX) - 1)
#define IF_MATCH_PREFIX "ifMatch="
#define IF_MATCH_PREFIX_LEN (sizeof(IF_MATCH_PREFIX) - 1)
#define IF_NOT_MATCH_PREFIX "ifNotMatch="
#define IF_NOT_MATCH_PREFIX_LEN (sizeof(IF_NOT_MATCH_PREFIX) - 1)
#define START_BYTE_PREFIX "startByte="
#define START_BYTE_PREFIX_LEN (sizeof(START_BYTE_PREFIX) - 1)
#define BYTE_COUNT_PREFIX "byteCount="
#define BYTE_COUNT_PREFIX_LEN (sizeof(BYTE_COUNT_PREFIX) - 1)
#define ALL_DETAILS_PREFIX "allDetails="
#define ALL_DETAILS_PREFIX_LEN (sizeof(ALL_DETAILS_PREFIX) - 1)
#define NO_STATUS_PREFIX "noStatus="
#define NO_STATUS_PREFIX_LEN (sizeof(NO_STATUS_PREFIX) - 1)
#define RESOURCE_PREFIX "resource="
#define RESOURCE_PREFIX_LEN (sizeof(RESOURCE_PREFIX) - 1)
#define TARGET_BUCKET_PREFIX "targetBucket="
#define TARGET_BUCKET_PREFIX_LEN (sizeof(TARGET_BUCKET_PREFIX) - 1)
#define TARGET_PREFIX_PREFIX "targetPrefix="
#define TARGET_PREFIX_PREFIX_LEN (sizeof(TARGET_PREFIX_PREFIX) - 1)
#define HTTP_METHOD_PREFIX "method="
#define HTTP_METHOD_PREFIX_LEN (sizeof(HTTP_METHOD_PREFIX) - 1)


typedef struct growbuffer
{
    // The total number of bytes, and the start byte
    int size;
    // The start byte
    int start;
    // The blocks
    char data[64 * 1024];
    struct growbuffer *prev, *next;
} growbuffer;

typedef struct put_object_callback_data
{
    FILE *infile;
    growbuffer *gb;
    uint64_t contentLength, originalContentLength;
    uint64_t totalContentLength, totalOriginalContentLength;
    int noStatus;
} put_object_callback_data;

void S3_init();
uint64_t convertInt(const char *str, const char *paramName);
int checkString(const char *str, const char *format);
int growbuffer_append(growbuffer **gb, const char *data, int dataLen);
void growbuffer_read(growbuffer **gb, int amt, int *amtReturn, char *buffer);
void growbuffer_destroy(growbuffer *gb);
int should_retry();
S3Status responsePropertiesCallback (const S3ResponseProperties *properties, void *callbackData);
void responseCompleteCallback(S3Status status, const S3ErrorDetails *error, void *callbackData);
int putObjectDataCallback(int bufferSize, char *buffer, void *callbackData);

