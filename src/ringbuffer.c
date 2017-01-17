#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdio.h>
// N must be 2^i
//#define N (16)

//void *b[N]

/* .raw read thread */
/* two ring buffers, one index */
/* 1: headers and 2: data */
/* write thread will throw a flag upon end of file */
/* read thread will simply grab next data chunk and process via memcpy */
/* ring buffer will be malloced in main thread */

#define RING_ELEMENTS (8)


struct gpu_input {
	char *file_prefix;
	struct guppi_params gf;
	struct psrfits pf;	
	unsigned int filecnt;
	FILE *fil;
	FILE *headerfile;
	int invalid;
	int curfile;
	int overlap;   /* add this keyword here since it doesn't seem to appear in guppi_params.c */
	long int first_file_skip; /* in case there's 8bit data in the header of file 0 */
	double baryv;
	double barya;
	unsigned int sqlid;
	

	struct hdrinfo * headers[RING_ELEMENTS];
	char * data[RING_ELEMENTS];
	int elements;
	int in;
	int out;
	pthread_mutex_t lock;
	sem_t countsem, spacesem;	
	
	
};




void enqueue(void *ptr);


struct input {
	float b[16];
	int N;
	int in;
	int out;
	pthread_mutex_t lock;
	sem_t countsem, spacesem;

};

int main () {

//pthread_join(accumwrite_th0, NULL);


pthread_t loadbuffer;

char buf[32768];

/*
struct input inputtest;
sem_t foo;
inputtest.N = 16;
inputtest.in = 0;
inputtest.out = 0;
pthread_mutex_init(&(inputtest.lock), NULL);
*/

  printf("init...%d \n", sem_init(&(inputtest.countsem), 0, 0));
  printf("init...%d \n", sem_init(&(inputtest.spacesem), 0, 16));
  printf("init...%d \n", sem_init(&foo, 0, 16));


pthread_create (&loadbuffer, NULL, (void *) &enqueue, (void *) &inputtest);

while(1) {
  // Wait if there are no items in the buffer
  sem_wait(&(inputtest.countsem));

  pthread_mutex_lock(&(inputtest.lock));
  printf("Popped: %f", inputtest.b[(inputtest.out++) & (inputtest.N-1)]);
  pthread_mutex_unlock(&(inputtest.lock));

  // Increment the count of the number of spaces
  sem_post(&(inputtest.spacesem));
  usleep(5000000);
}

return 0;
}





void enqueue(void *ptr){

struct input *inputtest;
inputtest = (struct input *) ptr;
int j = 0;
while(1) {
 // wait if there is no space left:
 printf("waiting...%d\n", sem_wait( &(inputtest->spacesem)));
 usleep(1000000);
 

 pthread_mutex_lock(&(inputtest->lock));
 inputtest->b[ (inputtest->in++) & (inputtest->N-1) ] = j;
 pthread_mutex_unlock(&(inputtest->lock));

 // increment the count of the number of items
 sem_post(&(inputtest->countsem));
 j++;
}
}

struct gpu_spectrometer *gpu_spec;
gpu_spec = (struct gpu_spectrometer *) ptr;

long int i,j,k;	 
	 
long int chanbytes_subint;
long int chanbytes_subint_total;
long int subint_cnt = 0;


chanbytes_subint = (gpu_spec->indxstep * gpu_spec->rawinput->pf.packetsize / ((8/gpu_spec->rawinput->pf.hdr.nbits) * gpu_spec->rawinput->pf.hdr.nchan));
chanbytes_subint_total = gpu_spec->rawinput->pf.sub.bytes_per_subint / gpu_spec->rawinput->pf.hdr.nchan;

/* all allocations in main() */
//channelbuffer  = (unsigned char *) calloc(gpu_spec->rawinput->chanbytes * gpu_spec->rawinput->pf.hdr.nchan, sizeof(unsigned char) );

for(i = 0; i < gpu_spec->rawinput->elements; i++) {
	gpu_spec->rawinput->data[i] = (unsigned char *) calloc(gpu_spec->rawinput->chanbytes * gpu_spec->rawinput->pf.hdr.nchan, sizeof(unsigned char) );	
	gpu_spec->rawinput->headers[i] = malloc(1, sizeof(struct hdrinfo));
}



//nchannels = gpu_spec->rawinput->pf.hdr.nchan;
//gpu_spec->triggerwrite




do{
							
										
	if(!gpu_spec->rawinput->invalid){						  
		  if(gpu_spec->rawinput->fil == NULL) {

			  /* no file is open for this band, try to open one */
			  sprintf(filname, "%s.%04d.raw",gpu_spec->rawinput->file_prefix,gpu_spec->rawinput->curfile);
			  printf("filename is %s\n",filname);
			  if(exists(filname)){
				 printf("opening %s\n",filname);				
				 gpu_spec->rawinput->fil = fopen(filname, "rb");			 

				 if(gpu_spec->rawinput->curfile == 0 && gpu_spec->rawinput->first_file_skip != 0) fseek(gpu_spec->rawinput->fil, gpu_spec->rawinput->first_file_skip, SEEK_CUR);  

			  }	else {
			  	gpu_spec->rawinput->invalid = 1;
		  	  	printf("couldn't open any more files!\n");
		  	  }
		  }

	if(gpu_spec->rawinput->fil){

		if(fread(buf, sizeof(char), 32768, gpu_spec->rawinput->fil) == 32768) {
				
			fseek(gpu_spec->rawinput->fil, -32768, SEEK_CUR);

			if(vflag>1) fprintf(stderr, "header length: %d\n", gethlength(buf));
			
			guppi_read_obs_params(buf, &gpu_spec->rawinput->gf, &gpu_spec->rawinput->pf);

			if(vflag>1) {
				 fprintf(stderr, "packetindex %Ld\n", gpu_spec->rawinput->gf.packetindex);
				 fprintf(stderr, "packetsize: %d\n", gpu_spec->rawinput->gf.packetsize);
				 fprintf(stderr, "n_packets %d\n", gpu_spec->rawinput->gf.n_packets);
				 fprintf(stderr, "n_dropped: %d\n",gpu_spec->rawinput->gf.n_dropped);
				 fprintf(stderr, "RA: %f\n",gpu_spec->rawinput->pf.sub.ra);
				 fprintf(stderr, "DEC: %f\n",gpu_spec->rawinput->pf.sub.dec);
				 fprintf(stderr, "subintoffset %f\n", gpu_spec->rawinput->pf.sub.offs);
				 fprintf(stderr, "tsubint %f\n", gpu_spec->rawinput->pf.sub.tsubint);
			}
					
				  if(gpu_spec->rawinput->gf.packetindex == curindx) {

					  /* read a subint with correct index, read the data */
					  if(gpu_spec->rawinput->pf.hdr.directio == 0){
						  hlength = (long int) gethlength(buf);

						  /* write out header for archiving */
						  fwrite(buf, sizeof(char), hlength, gpu_spec->rawinput->headerfile);

						  fseek(gpu_spec->rawinput->fil, gethlength(buf), SEEK_CUR);
						  rv=fread(gpu_spec->rawinput->pf.sub.data, sizeof(char), gpu_spec->rawinput->pf.sub.bytes_per_subint, gpu_spec->rawinput->fil);		 
 
								//lseek(filehandle, gethlength(buf), SEEK_CUR);				
 								//rv = read(filehandle, gpu_spec->rawinput->pf.sub.data, gpu_spec->rawinput->pf.sub.bytes_per_subint);
					  } else {
					  	  hlength = (long int) gethlength(buf);
					  	  
					  	  /* write out header for archiving */
						  fwrite(buf, sizeof(char), hlength, gpu_spec->rawinput->headerfile);

					  	  if(vflag>1) fprintf(stderr, "header length: %ld\n", hlength);
						  if(vflag>1) fprintf(stderr, "seeking: %ld\n", hlength + ((512 - (hlength%512))%512) );
					  	  fseek(gpu_spec->rawinput->fil, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);
							//lseek(filehandle, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);				

						    rv=fread(gpu_spec->rawinput->pf.sub.data, sizeof(char), gpu_spec->rawinput->pf.sub.bytes_per_subint, gpu_spec->rawinput->fil);

 							//rv = read(filehandle, gpu_spec->rawinput->pf.sub.data, gpu_spec->rawinput->pf.sub.bytes_per_subint);

						  fseek(gpu_spec->rawinput->fil, ( (512 - (gpu_spec->rawinput->pf.sub.bytes_per_subint%512))%512), SEEK_CUR);
						 //lseek(filehandle, ( (512 - (gpu_spec->rawinput->pf.sub.bytes_per_subint%512))%512), SEEK_CUR);				

					  }
					  
					  if((long int)rv == gpu_spec->rawinput->pf.sub.bytes_per_subint){
						 if(vflag>1) fprintf(stderr,"read %d bytes from %ld in curfile %d\n", gpu_spec->rawinput->pf.sub.bytes_per_subint, j, gpu_spec->rawinput->curfile);
						  
						 /* need to have each channel be contiguous */
						 /* copy in to buffer by an amount offset by the total channel offset + the offset within that channel */
						 /* need to make sure we only grab the non-overlapping piece */
						 for(i = 0; i < gpu_spec->rawinput->pf.hdr.nchan; i++) {
							 memcpy(channelbuffer + (i * chanbytes) + (subint_cnt * chanbytes_subint), gpu_spec->rawinput->pf.sub.data + (i * chanbytes_subint_total), chanbytes_subint);												
						 }
						 //memcpy(channelbuffer + (subint_cnt * gpu_spec->rawinput->pf.sub.bytes_per_subint), gpu_spec->rawinput->pf.sub.data, gpu_spec->rawinput->pf.sub.bytes_per_subint);
						 subint_cnt++;
			  
			  
						 if(vflag>=1) fprintf(stderr, "copied %lld bytes subint cnt %ld\n", chanbytes * gpu_spec->rawinput->pf.hdr.nchan, subint_cnt);
			  
			
					   } else {
						   gpu_spec->rawinput->fil = NULL;
						   gpu_spec->rawinput->invalid = 1;
						   fprintf(stderr,"ERR: couldn't read as much as the header said we could... assuming corruption and exiting...\n");
						   exit(1);
					   }
				   
			  
				   } else if(gpu_spec->rawinput->gf.packetindex > curindx) {
						fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, gpu_spec->rawinput->gf.packetindex );
						/* read a subint with too high an indx, must have dropped a whole subintegration*/
						/* don't read the subint, but increment the subint counter and allow old data to be rechannelized */
						/* so that we maintain continuity in time... */
						subint_cnt++;
						//curindx = curindx + indxstep;
					   /* We'll get the current valid subintegration again during the next time through this loop */

			  
				   } else if(gpu_spec->rawinput->gf.packetindex < curindx) {
						fprintf(stderr,"ERR: curindx: %Ld, pktindx: %Ld\n", curindx, gpu_spec->rawinput->gf.packetindex );
						/* somehow we were expecting a higher packet index than we got !?!? */

						/* we'll read past this subint and try again next time through */

						 if(gpu_spec->rawinput->pf.hdr.directio == 0){
							 fseek(gpu_spec->rawinput->fil, gethlength(buf), SEEK_CUR);
							 rv=fread(gpu_spec->rawinput->pf.sub.data, sizeof(char), gpu_spec->rawinput->pf.sub.bytes_per_subint, gpu_spec->rawinput->fil);		 
 
								   //lseek(filehandle, gethlength(buf), SEEK_CUR);				
								   //rv = read(filehandle, gpu_spec->rawinput->pf.sub.data, gpu_spec->rawinput->pf.sub.bytes_per_subint);
						 } else {
							 hlength = (long int) gethlength(buf);
							 if(vflag>1) fprintf(stderr, "header length: %ld\n", hlength);
							 if(vflag>1) fprintf(stderr, "seeking: %ld\n", hlength + ((512 - (hlength%512))%512) );
							 fseek(gpu_spec->rawinput->fil, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);
							  //lseek(filehandle, (hlength + ((512 - (hlength%512))%512) ), SEEK_CUR);				

							 rv=fread(gpu_spec->rawinput->pf.sub.data, sizeof(char), gpu_spec->rawinput->pf.sub.bytes_per_subint, gpu_spec->rawinput->fil);

							   //rv = read(filehandle, gpu_spec->rawinput->pf.sub.data, gpu_spec->rawinput->pf.sub.bytes_per_subint);

							 fseek(gpu_spec->rawinput->fil, ( (512 - (gpu_spec->rawinput->pf.sub.bytes_per_subint%512))%512), SEEK_CUR);
							  //lseek(filehandle, ( (512 - (gpu_spec->rawinput->pf.sub.bytes_per_subint%512))%512), SEEK_CUR);				

						 }
						 
						 curindx = curindx - indxstep;

				   }

				   if(subint_cnt == gpu_spec->rawinput->num_bufs) {
				   			subint_cnt=0;
				   			
							// wait if there is no space left:
							sem_wait( &(gpu_spec->rawinput->spacesem));
							
							//lock the mutex
							pthread_mutex_lock(&(gpu_spec->rawinput->lock));
							
							memcpy(data[(gpu_spec->rawinput->in++) & (gpu_spec->rawinput->elements - 1)], channelbuffer, (size_t) nsamples * nchannels * 4);
							memcpy(headers[(gpu_spec->rawinput->in++) & (gpu_spec->rawinput->elements - 1)], gpu_spec->rawinput->pf.hdr, sizeof(struct hdrinfo));
							
						    // Unlock the mutex
							pthread_mutex_unlock(&(gpu_spec->rawinput->lock));

							// increment the count of the number of items
							sem_post(&(inputtest->countsem));
				   }




			   } else {

			   /* file open but couldn't read 32KB */
				  fclose(gpu_spec->rawinput->fil);
				  gpu_spec->rawinput->fil = NULL;
				  //close(filehandle);
				  //filehandle=-1;
				  gpu_spec->rawinput->curfile++;						
			   }
		}			 	 	 
	}

										
	if(gpu_spec->rawinput->fil != NULL) curindx = curindx + indxstep;
//	if(filehandle > 0) curindx = curindx + indxstep;


} while(!(gpu_spec->rawinput->invalid));