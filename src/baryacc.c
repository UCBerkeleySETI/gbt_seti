#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include "barycenter.h"


int main(int argc, char *argv[]) {

	double topotimes[2];
	double barytimes[2];
	double voverc[2];
	char obscode[4];
	char ephemcode[8];

	/*hard code for GB */
	strcpy(obscode, "GB");
	strcpy(ephemcode, "DE405");


	char *rastr;
	char *decstr;

	double mjd;
	double timespan;

	if(argc < 2) {
		exit(1);
	}

	int c;
	long int i,j,k;
	opterr = 0;
 
	while ((c = getopt (argc, argv, "hr:d:m:t:")) != -1)
	  switch (c)
		{
		case 'h':
		  printf(" -r <ra string>\n");
		  printf(" -d <dec string>\n");
		  printf(" -m <MJD> \n");
		  printf(" - t <timespan in seconds> \n");
		  printf("\n");
		  printf("\n");
		  printf("\n");
		  
		  exit(0);
		  break;
		case 'r':
		  rastr = optarg;
		  break;
		case 'd':
		  decstr = optarg;
		  break;
		case 'm':
		  mjd = (float) atof(optarg);
		  break;
		case 't':
		  timespan = (float) atof(optarg);
		  break;
		case '?':
		  if (optopt == 'i' || optopt == 'o' || optopt == '1' || optopt == '2' || optopt == '3' || optopt == '4' || optopt == '5' || optopt == '6'|| optopt == '7' || optopt == '8')
			fprintf (stderr, "Option -%c requires an argument.\n", optopt);
		  else if (isprint (optopt))
			fprintf (stderr, "Unknown option `-%c'.\n", optopt);
		  else
			fprintf (stderr,
					 "Unknown option character `\\x%x'.\n",
					 optopt);
		  return 1;
		default:
		  abort ();
		}
		



/* barycentric velocity correction */
/* barycentric acceleration */



topotimes[0] = mjd - (timespan/2)/SECPERDAY;
topotimes[1] = mjd + (timespan/2)/SECPERDAY;



barycenter(topotimes, barytimes, voverc, 2, rastr, decstr, obscode, ephemcode);

double baryv = voverc[0];
double barya = (voverc[0]-voverc[1])/timespan;

printf("start time %15.15g end time %15.15g %s %s barycentric velocity %15.15g barycentric acceleration %15.15g \n", mjd - (timespan/2)/SECPERDAY, mjd + (timespan/2)/SECPERDAY, rastr, decstr, baryv, barya);


}