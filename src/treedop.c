#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>

float randomFloat()
{
      float r = (float)rand() / (float)RAND_MAX;
      return r;
}


int     DMMin, DMMax, FlipSwitch, FlipSwitch;
double  TSkip, TRead, DMMinv, DMMaxv;
char   *unfname;

int nbands, nobits;

/*
void tree_help() {
  puts("");
  puts("tree - dedisperses filterbank data rapidly using the tree algorithm");
  puts("");
  puts("usage : tree <options> <UniqueID>");
  puts("");
  puts("-s skip - time length to skip (s; def=0)");
  puts("-r read - time length to read (s; def=largest 2^n*tsamp)");
  puts("-l dmlo - lower DM index to write (def=0)");
  puts("-u dmup - upper DM index to write (def=nchan-1)");
  puts("");
  exit(0);
}
*/

/*  ======================================================================  */
/*  This function bit-reverses the given value "inval" with the number of   */
/*  bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.               */
/*  ======================================================================  */


int bitrev(int inval,int nbits)
{
     int     ifact,k,i,ibitr;

     if(nbits <= 1)
     {
          ibitr = inval;
     }
     else
     {
          ifact = 1;
          for (i=1; i<(nbits); ++i)
               ifact  *= 2;
          k     = inval;
          ibitr = (1 & k) * ifact;

          for (i=2; i < (nbits+1); i++)
          {
               k     /= 2;
               ifact /= 2;
               ibitr += (1 & k) * ifact;
          }
     }
     return ibitr;
}

void AxisSwap(float Inbuf[],
              float Outbuf[], 
              int   nchans, 
              int   NTSampInRead) {
  int    j1, j2, indx, jndx;

  for (j1=0; j1<NTSampInRead; j1++) {
    indx  = (j1 * nchans);
    for (j2=(nchans-1); j2>=0; j2--) {
      jndx = (j2 * NTSampInRead + j1);
      Outbuf[jndx]  = Inbuf[indx+j2];
    }
  }

  return;
}
void  FlipBand(float  Outbuf[], 
               int    nchans, 
               int    NTSampInRead) {

  int    indx, jndx, kndx, i, j;
  float *temp;

  temp  = (float *) calloc((NTSampInRead*nchans), sizeof(float));

  indx  = (nchans - 1);
  for (i=0; i<nchans; i++) {
    jndx = (indx - i) * NTSampInRead;
    kndx = (i * NTSampInRead);
    memcpy(&temp[jndx], &Outbuf[kndx], sizeof(float)*NTSampInRead);
  }
  memcpy(Outbuf, temp, (sizeof(float)*NTSampInRead * nchans));

  free(temp);

  return;
}

/*  ======================================================================  */
/*  This is a function to Taylor-dedisperse a data stream. It assumes that  */
/*  the arrangement of data stream is, all points in Chn.1, all points in   */
/*  Chn.2, and so forth.                                                    */
/*                     R. Ramachandran, 07-Nov-97, nfra.                    */
/*                                                                          */
/*  outbuf[]       : input array (short int), replaced by dedispersed data  */
/*                   at the output                                          */
/*  mlen           : dimension of outbuf[] (int)                            */
/*  nchn           : number of frequency channels (int)                     */
/*                                                                          */
/*  ======================================================================  */

void taylor_flt(float outbuf[], int mlen, int nchn)
{
  float itemp;
  int   nsamp,npts,ndat1,nstages,nmem,nmem2,nsec1,nfin, i;
  int   istages,isec,ipair,ioff1,i1,i2,koff,ndelay,ndelay2;
  int   bitrev(int, int);

  /*  ======================================================================  */

  nsamp   = ((mlen/nchn) - (2*nchn));
  npts    = (nsamp + nchn);
  ndat1   = (nsamp + 2 * nchn);
  
  //nstages = (int)(log((float)nchn) / 0.6931471 + 0.5);
  nstages = (int) log2f((float)nchn);
  nmem    = 1;


  for (istages=0; istages<nstages; istages++) {
    nmem  *= 2;
    nsec1  = (nchn/nmem);
    nmem2  = (nmem - 2);

    for (isec=0; isec<nsec1; isec++) {
      ndelay = -1;
      koff   = (isec * nmem);

      for (ipair=0; ipair<(nmem2+1); ipair += 2) {
        

        ioff1   = (bitrev(ipair,istages+1)+koff) * ndat1;
        i2      = (bitrev(ipair+1,istages+1) + koff) * ndat1;
        ndelay++;
        ndelay2 = (ndelay + 1);
        nfin    = (npts + ioff1);
        for (i1=ioff1; i1<nfin; i1++) {
          itemp      = (outbuf[i1] + outbuf[i2+ndelay]);
          outbuf[i2] = (outbuf[i1] + outbuf[i2+ndelay2]);
          outbuf[i1] = itemp;
          i2++;

        }
      }
    }
  }

  return;
}


main(int argc, char** argv)
{


fftwf_complex *in;
fftwf_complex *out;
fftwf_plan plan_forward;

int i,j,k;
float *tree_dedisperse;
int *ibrev;
int indx;
int xdim = 512;
int ydim = 1048576;
tree_dedisperse = (float*) malloc(xdim * ydim * sizeof(float));
/*
for(i = 0; i< 512;i++){
	tree_dedisperse[i]=(float*) malloc(ydim * sizeof(float));
}
*/
fftwf_import_wisdom_from_file("/home/siemion/sw/kepler/treedop_wisdom.txt");
in = fftwf_malloc ( sizeof ( fftwf_complex ) * ydim );
out = fftwf_malloc ( sizeof ( fftwf_complex ) * ydim );
plan_forward = fftwf_plan_dft_1d ( ydim, in, out, FFTW_FORWARD, FFTW_PATIENT);
//fftwf_export_wisdom_to_filename("/home/siemion/sw/kepler/treedop_wisdom.txt");


printf("loading values...\n");fflush(stdout);

for(i=0;i<xdim;i++){
   
   for(j=0;j<ydim;j++){
		tree_dedisperse[(i*ydim) + j] = randomFloat(); 
		in[j][0] = tree_dedisperse[(i*ydim) + j];
		in[j][1] = tree_dedisperse[(i*ydim) + j];
		//printf("%f, ", tree_dedisperse[i*xdim+j]);
   }
   fftwf_execute ( plan_forward );

	//printf("\n");
}
printf("done...\n");fflush(stdout);

	printf("\n\n");

ibrev = (int *) calloc(xdim, sizeof(int));

for (i=0; i<xdim; i++) {
    ibrev[i] = bitrev(i, (int) log2f(xdim));
}

/*
tree_dedisperse[0]=0;
tree_dedisperse[8]=0;
tree_dedisperse[16]=0;
tree_dedisperse[24]=0;


tree_dedisperse[4]=1;
tree_dedisperse[ydim+4]=2;
tree_dedisperse[ydim*2 + 4]=3;
tree_dedisperse[ydim * 3 + 4]=4;
*/

printf("dedispersing...\n");fflush(stdout);

taylor_flt(tree_dedisperse, xdim * ydim, xdim);
 printf("done...\n");fflush(stdout);


//fwrite(&Outbuf[indx], sizeof(float), MAX, Fout[iw-DMMin]);

 for(i=0;i<xdim;i++){
   for(j=0;j<ydim;j++){
		indx  = (ibrev[i] * ydim);
		//if(tree_dedisperse[indx+j] != 0) printf("%02i: %02f, ", (i * ydim+j)%ydim, tree_dedisperse[indx+j]);
   }
	//printf("\n");
}
/*
for(i=0;i<span/2;i++){
out[i] = tree[i][j] + 

out[0] = tree[0][i] + tree[1][i]
out[1] = tree[0][i] + tree




*/
//channel_r = rand(4130816)
//channel_i = rand(4130816);
  exit(0);
}
