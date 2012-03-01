struct gpu_input {
	char *file_prefix;
	struct guppi_params gf;
	struct psrfits pf;	
	unsigned int filecnt;
	FILE *fil;
	int invalid;
	int curfile;
	int overlap;   /* add this keyword here since it doesn't seem to appear in guppi_params.c */
	long int first_file_skip; /* in case there's 8bit data in the header of file 0 */
	double baryv;
	double barya;
	unsigned int sqlid;
};

struct max_vals {
	float *maxsnr;
	float *maxdrift;
	unsigned char *maxsmooth;
	unsigned long int *maxid;
};

struct fftinfo {

	float *spectra;
	unsigned char *channelbuffer;
	long int numsamples;
	long int fftlen;
	fftwf_complex *in;
	fftwf_complex *out;
	fftwf_plan plan_forward;
	float *bandpass;

};


long int candsearch(float * spectrum, long int specstart, long int specend, int candthresh, float drift_rate, \
		struct gpu_input * firstinput, long int fftlen, long int tdwidth, long int channel, struct max_vals * max, unsigned char reverse);
		
void simple_fits_buf(char * fitsdata, float *vec, int height, int width, double fcntr, long int fftlen, double snr, double doppler, struct gpu_input *firstinput);
		

void comp_stats(double *mean, double *stddev, float *vec, long int veclen, char *ignore);
