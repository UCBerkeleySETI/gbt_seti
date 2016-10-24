%
% Contents file for package: sdss
% Created: 29-Dec-2015
%---------
% coo2run.m :  Given celestial equatorial coordinates, find all SDSS Run/Rerun/Col/Field number ID that cover the coordinates.
% find_all_sdss82.m :  Look for all SDSS strip 82 images within a given box region. This script search the SDSS82_Fields.mat file
% get_all_sdss82.m :  Download all SDSS strip 82 images within a given box region, in a given JD range and with a given seeing.
% get_sdss_corrim.m :  Given an SDSS field ID [Run, Rerun, Camcol, Field], get the link to, and the SDSS corrected image in FITS format. Furthermore, read the corrected image into matlab matrix. The program first check if the FITS image is exist in current directory and if so, it reads the image from the disk. Note that if nargout>1 then the fits file is retrieved.
% get_sdss_finding_chart.m :  Get a link, JPG (and read it into matlab) of an SDSS finding chart directly from the SDSS sky server (Version: SDSS-DR8).
% get_sdss_imcat.m :  Given an SDSS field ID [Run, Rerun, Camcol, Field], get the link to, and the SDSS object catalog (tsObj) associated with a corrected image in FITS format. Furthermore, read the catalog into a matlab matrix. The program first check if the FITS table exist in current directory and if so, it reads the image from the disk. Note that if nargout>1 then the fits file is retrieved.
% get_sdss_spectra.m :  Given an SDSS soectra ID [Plate, MJD, Fiber], get the link to, and the SDSS 1d spectra in FITS format. Furthermore, read the spectra into matlab matrix. The program first check if the FITS image is exist in current directory and if so, it reads the image from the disk. Note that if nargout>1 then the fits file is retrieved.
% get_sdss_tsfield.m :  Given an SDSS field ID [Run, Rerun, Camcol, Field], get the link to, and the SDSS field catalog information (tsField) associated with a corrected image in FITS format. Furthermore, read the catalog into a matlab matrix. The program first check if the FITS table exist in current directory and if so, it reads the image from the disk. Note that if nargout>1 then the fits file is retrieved.
% mag_calib_tsfield.m :  Use (or get and use) the SDSS tsField fits file to calculate the photometric zero point for a set of an SDSS FITS images.
% prep_stripe82_deep.m :  Download all sdss stripe 82 images within a given box, a given JD range and seeing, and combine the images using swarp. The ouput images are named coadd.fits and coadd.weight.fits.
% read_sdss_spec.m :  Read SDSS spectra in FITS format into matlab structure.
% run2coo.m :  Convert SDSS run/rerun/camcol/field/object ID to coordinates.
% run_sdss_sql.m :  Run SQL query on SDSS database and retrieve the results into an array.
% sdss_coo_radec.m :  Convert the SDSS great circles coordinate system to J2000 right ascension and declination.
% sdss_spec_mask_select.m :  Given a mask of the SDSS spectra in decimal format return indices for all the masks which are good and bad. Good mask defined to have "SP_MASK_OK" | "SP_MASK_EMLINE".
