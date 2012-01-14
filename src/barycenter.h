#ifndef SECPERDAY
#define SECPERDAY     86400.0
#endif

double doppler(double freq_observed, double voverc);
  /* This routine returns the frequency emitted by a pulsar */
  /* (in MHz) given that we observe the pulsar at frequency */
  /* freq_observed (MHz) while moving with radial velocity  */
  /* (in units of v/c) of voverc wrt the pulsar.            */


void barycenter(double *topotimes, double *barytimes, double *voverc, long N, char *ra, char *dec, char *obs, char *ephem);

  /* This routine uses TEMPO to correct a vector of           */
  /* topocentric times (in *topotimes) to barycentric times   */
  /* (in *barytimes) assuming an infinite observation         */
  /* frequency.  The routine also returns values for the      */
  /* radial velocity of the observation site (in units of     */
  /* v/c) at the barycentric times.  All three vectors must   */
  /* be initialized prior to calling.  The vector length for  */
  /* all the vectors is 'N' points.  The RA and DEC (J2000)   */
  /* of the observed object are passed as strings in the      */
  /* following format: "hh:mm:ss.ssss" for RA and             */
  /* "dd:mm:s.ssss" for DEC.  The observatory site is passed  */
  /* as a 2 letter ITOA code.  This observatory code must be  */
  /* found in obsys.dat (in the TEMPO paths).  The ephemeris  */
  /* is either "DE200" or "DE400".                            */
  
