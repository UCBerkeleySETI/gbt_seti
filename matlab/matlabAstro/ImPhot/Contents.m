%
% Contents file for package: ImPhot
% Created: 29-Dec-2015
%---------
% aperphot.m :  Aperture photometry. Given a matrix and a list of coordinates, calculate accurate centers for each object and perform aperture photometry.
% def_bitmask_photpipeline.m :  The spectroscopic pipeline bit mask definition. Given the Bit mask name return the bit mask index.
% flux2mag.m :  Convert flux to magnitude or luptitude. 
% hst_acs_zp_apcorr.m :  Given aperture radius for photometry, and Hubble Space Telecsope (HST) ACS-WFC filter name, return the fraction, and error in fraction, of the encircled energy within the aperture.
% mextractor.m :  Source extractor written in MATLAB. Given an input images, and matched filter, this function filter the images by their matched filter and search for sources above a detection threshold. For each source the program measure its basic properties. The main difference between this program and SExtractor is that the noise ("sigma") is measured in the filtered image, rather than the unfiltered image.
% moment2d.m :  Given an image, calculate the moments and flux around given set of coordinates.
% optimal_phot_aperture.m :  Given a Gaussian symmetric PSF and image background and readout noise. Estimate the radius of the photometric aperture that maximize the S/N for an aperture photometry, for a given target S/N. Return also the signal of the source. Calculate under the assumption that the number of pixels in the background is continuous and not discrete. Also calculate the S/N for a PSF photometry.
% phot_relzp.m :  Given a matrix of instrumental magnitudes (and possibly errors) for aech epoch (image) and each source find the best fit zero point per epoch and mean magnitude per star that will minimize the residuals of the sources over all epochs. In its most basic form, this function solve the equation: InstMag_ij = ZP_i + <Mag>_j, where ZP_i is the zero-point of the i-th image, and <Mag>_j is the mean magnitude of the j-th star. UNDER DEVELOPMEMT - MAY CHANGE.
% psf_builder.m :  Construct the PSF for an images by averaging selected stars.
% query_columns.m :  Query a table or catalog using a string containing logical operations on column names.
% run_sextractor.m :  Run SExtractor from matlab. This function is obsolte. Use sextractor.m instead.
% sextractor.m :  execute sextracor on a set of FITS images or SIM images.
% simcat_matchcoo.m :  Given a structure array (SIM) of images and source catalogs and a reference image/catalog, match by coordinates the sources in each image against the reference catalog. For each requested property (e.g., 'XWIN_IMAGE') in the catalog, the function returns a matrix of the property of the matched sources. The matrix (ImageIndex,SourceIndex), rows and columns corresponds to images, and sources in the reference image, respectivelly. In addition, the function returns a structure array (element per image) of un-matched sources.
% sn_aper_phot.m :  Calculate the S/N (signal-to-noise ratio) for a point source with a symmetric Gaussian profile for aperture photometry.
% sn_det2psf_signal.m :  Given detection S/N calculate the PSF signal.
% sn_psf_det.m :  Calculate the S/N (signal-to-noise ratio) for a point source with a symmetric Gaussian profile for PSF (optimal) detection. Note this is different than PSF photometry (see sn_psf_phot.m).
% sn_psf_phot.m :  Calculate the S/N (signal-to-noise ratio) for a point source with a symmetric Gaussian profile for PSF (optimal) photometry.
% sn_psfn_phot.m :  Calculate the S/N (signal-to-noise ratio) for a numerical PSF (optimal) photometry.
% sum_mag.m :  Sum a set of magnitudes.
