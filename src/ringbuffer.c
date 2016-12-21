#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
// N must be 2^i
//#define N (16)

//void *b[N]

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


struct input inputtest;
sem_t foo;
inputtest.N = 16;
inputtest.in = 0;
inputtest.out = 0;
pthread_mutex_init(&(inputtest.lock), NULL);

  printf("init...%d \n", sem_init(&(inputtest.countsem), 0, 0));
  printf("init...%d \n", sem_init(&(inputtest.spacesem), 0, 16));
  printf("init...%d \n", sem_init(&foo, 0, 16));


pthread_create (&loadbuffer, NULL, (void *) &enqueue, (void *) &inputtest);

while(1) {
  // Wait if there are no items in the buffer
  sem_wait(&(inputtest.countsem));

  pthread_mutex_lock(&(inputtest.lock));
  printf("Popped: %d", b[(out++) & (N-1)]);
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

