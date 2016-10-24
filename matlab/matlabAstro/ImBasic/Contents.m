%
% Contents file for package: ImBasic
% Created: 29-Dec-2015
%---------
% addcat2sim.m :  Add a catalog into structure image array (SIM) or upload images and catalogs into SIM. The catalogs can be provided as a FITS table, or extracted from images.
% bias_construct.m :  Construct a bias (or dark) image from a set bias (or dark) images. This function is not responsible for selecting good bias images.
% ccdsec_convert.m :  Convert CCDSEC format (e.g., '[1:100,201:301]') from string to vector and vise versa.
% cell_fitshead_addkey.m :  A utility program to add new keywords, values and comments to a cell array containing A FITS header information. The FITS header cell array contains an arbitrary number of rows and 3 columns, where the columns are: {keyword_name, keyword_val, comment}. The comment column is optional.
% cell_fitshead_delkey.m :  A utility program to delete a keywords, values and comments from a cell array containing A FITS header information. The FITS header cell array contains an arbitrary number of rows and 3 columns, where the columns are: {keyword_name, keyword_val, comment}. The comment column is optional.
% cell_fitshead_fix.m :  Given an Nx3 cell array of FITS header. Remove blank lines and make sure the END keyword is at the end of the header.
% cell_fitshead_getkey.m :  A utility program to get a specific keywords, values and comments from a cell array containing A FITS header information. The FITS header cell array contains an arbitrary number of rows and 3 columns, where the columns are: {keyword_name, keyword_val, comment}. The comment column is optional.
% cell_fitshead_search.m :  Search for substring in FITS header stored as a cell array of 3 columns. Return and display all instances of the substring.
% cell_fitshead_update.m :  Update keywords in fits header cell.
% clip_image_mean.m :  Given a cube of images (image index is the third dimension), calculate the sigma clipped mean/median.
% col_name2ind.m :  Convert a columns index structure to cell of column names, and convert specific column names to indices.
% col_name2indvec.m :  Given a cell array of names and another cell array containing a subset of column names, return the indices of the subset columns in the superset cell array.
% construct_matched_filter2d.m :  Construct an optimal 2D matched filter for sources detection in an image (given by S/(N^2)).
% conv2_gauss.m :  Convolve an image with a Gaussian or a top hat.
% create_mat3d_images.m :  Given a list of FITS images, read them into a 3D matrix, in which the third axis corresponds to the image index.
% cube2sim.m :  Convert a cube to a structure array of images (SIM).
% cut_image.m :  Cut subsection of an image (matrix) given the subsection coordinates, or center and size of subsection.
% filter2smart.m :  Cross correlate (i.e., filter) a 2D image with a given filter. The function is using either filter2.m or direct fft, according to the filter size compared with the image size.
% fits_delete_keywords.m :  Delete a list of header keywords from a list of FITS images.
% fits_get_head.m :  Read a specific Header Data Unit (HDU) in a FITS file into a cell array of {Keyword, Value, comment}.
% fits_get_keys.m :  Get the values of specific keywords from a single FITS file header. Use only for existing keywords.
% fits_header_cell.m :  Read FITS image header to cell header. This function can be used instead of fitsinfo.m This function is obsolte: use fits_get_head.m instead.
% fits_mget_keys.m :  Get the values of specific keywords from a list of FITS files header.
% fits_write_keywords.m :  Insert new, or update existing FITS header keywords in a list of FITS images.
% fitshead.m :  Read FITS file header, convert it to a string, and print it to screen. Supress empty header lines.
% fitsread_section.m :  Read a rectangular region of interest from a single FITS image.
% fitsread_section_many.m :  Read a rectangular region of interest from a single FITS image.
% fitswrite.m :  Write a simple 2D FITS image. USE fitswrite_my.m instead!
% fitswrite_me.m :  Write a set of images to a single multi-extension FITS file.
% fitswrite_my.m :  Write a simple 2D FITS image.
% fitswrite_my1.m :  Write a simple 2D FITS image.
% fitswrite_nd.m :  Write a multi-dimensional FITS image to a file.
% fitswrite_opt.m :  Write a FITS image using fitswrite_my.m. This a version fitswrite_my.m with some additional options.
% flat_construct.m :  Construct a flat field image from a set of images. This function is not responsible for selecting good flat images.
% get_bitmask_def.m :  A database of bit mask definitions for astronomical images. This define the default bit mask definitions.
% get_ccdsec_head.m :  Get an parse CCD section keyword value from a cell array containing an image header or a structure array containing multiple image headers.
% get_fits_keyword.m :  Get list of user selected keywords value from the header of a single FITS file. See also: mget_fits_keyword.m
% get_fitstable_col.m :  Read a FITS table, and get the fits table column names from the FITS header.
% get_sextractor_segmentation.m :  Given a list of FITS images create segmentation FITS images for each one of them using SExtractor.
% image2sim.m :  Read a single image to structure image data (SIM). For multiple files version of this program see images2sim.m.
% image_art.m :  Generate artificial images or add artificial sources to real images.
% image_background.m :  Generate a background image and also a background subtracted image.
% image_binning.m :  Bin an image (whole pixel binning)
% image_noise.m :  Calculate the teoretical noise image of a real image using its gain and readout noise.
% image_shift.m :  Shift an image in X and Y.
% images2sim.m :  Read multiple images to structure array image data (SIM). For single file version of this program see image2sim.m.
% imcombine.m :  Combine set of 2D arrays into a single 2D array. This function use imcombine_fits.m to combine FITS images.
% imcrdetect.m :  Find and remove cosmic rays in an astronomical image using the L.A.cosmic algorithm (van Dokkum 2001).
% iminterp.m :  In a 2-D image interpolate over NaNs or over pixels in which a bit mask i set.
% imlaplacian.m :  Calculate the laplacian of a 2-D image using a convolution kernel. This function is considerably faster than del2.m
% is_arc_image.m :  Given a list of FITS images or SIM, look for arc (wavelength calibration) images. The search is done by looking for specific keyword values in the image headers.
% is_bias_image.m :  Given a list of FITS images or SIM, look for good bias images. The search is done by looking for specific keyword values in the image headers, and also by checking the noise and mean properties of the images.
% is_flat_image.m :  Given a list of FITS images or SIM, look for good flat field images. The search is done by looking for specific keyword values in the image headers.
% is_head_keyval.m :  Given image headers in a 3 column cell array format, go over a list of keywords and check if their values are equal to some specific strings or numbers.
% is_saturated_image.m :  Count how many saturated pixels are in each image.
% issim.m :  Check if object is of SIM class.
% lineprof.m :  Given two coordinates in a matrix the script return the intensity as function of position along the line between the two points.
% medfilt2nan.m :  2-D median filter that ignores NaN's. This is similar to medfilt2.m, but ignoring NaNs.
% medfilt_circ.m :  Run a 2-D circular median filter (ignoring NaN) on an image. This function is very slow.
% mget_fits_keyword.m :  Get list of user selected keywords value from the headers of multiple FITS files. See also get_fits_keyword.m
% mode_image.m :  Calculate the mode value of a 1D or 2D matrix. This function use bins widths which are adjusted to the noise statistics of the image.
% prep_output_images_list.m :  Prepare the output images list by concanating the output directory and prefix to file names. This is a utility program mainly used by some of the *_fits.m functions.
% read2sim.m :  Read a list of images into a structure array. A flexible utility that allows to generate a structure array of images from one of the following inputs: a matrix; cell of matrices; a string with wild cards (see sdir for flexibility); a file containing list of images (see create_list.m); a cell array of file names; or a structure array with image names. see read2cat.m for the analog function that works with catalogs. THIS FUNCTION IS OBSOLOTE use images2sim.m instead.
% read_fitstable.m :  Read binary or ascii FITS tables.
% sim2cube.m :  Given a structure array of images (generated by read2sim.m), generate a cube of all the images, in which the first dimension is the image index.
% sim2file.m :  Given a structure array of images (e.g., that was constructed by read2im.m), write each one of the images to the disk in several possible formats.
% sim2fits.m :  Write a single elemnet of a structure image array (SIM) as a FITS file. See sims2fits.m for writing multiple files.
% sim_back_std.m :  Estimate background and std of an image. The background and std are calculated in local blocks (bins), and than interpolated to each pixel.
% sim_background.m :  Generate a background image and also a background subtracted image.
% sim_bias.m :  Given a set of images, construct a bias image (or use a user supplied bias image), subtract the bias image from all the science images and save it to a structure array of bias subtracted images.
% sim_ccdsec.m :  Get CCDSEC keyword value from a Sim structure array. If not available use image size.
% sim_class.m :  Convert SIM image class to another class.
% sim_coadd.m :  Image caoddition, with optional offset, scaling, weighting and filtering.
% sim_coadd_proper.m :  Image caoddition, with optional offset, scaling, weighting and filtering.
% sim_combine.m :  Combine set of 2D arrays into a single 2D array. This function use imcombine_fits.m to combine FITS images.
% sim_conv.m :  Convolve a set of a structure images with a kernel.
% sim_crdetect.m :  For images stored in a structure array (SIM) - Find and remove cosmic rays in an astronomical image using the L.A.cosmic algorithm (van Dokkum 2001).
% sim_fft.m :  Calculate FFT or inverse FFT of structure images array (or SIM class).
% sim_filter.m :  cross-correlate SIM images with filters.
% sim_flat.m :  Given a set of images, construct a flat image (or use a user supplied flat image), correct all the images by the flat image and save it to a structure array of flat field corrected images.
% sim_flip.m :  Flip or transpose a set of structure images (SIM).
% sim_gain.m : NaN
% sim_get_stamp.m :  Get image stamps around locations from a set of images.
% sim_getkeyval.m :  Give a structure array of images and their headers, get the values of a specific header keyword. See sim_getkeyvals.m for a multiple keyword version.
% sim_getkeyvals.m :  Give a structure array of images and their headers, get the values of a specific header keywords.
% sim_group_keyval.m :  Given a structure array with image headers, and a list of header keywords, construct groups of images with identical keyword header values.
% sim_head_search.m :  Search for substring in Sim/FITS image headers.
% sim_imagesize.m :  Get the image size of a set of images.
% sim_imarith.m :  Perform a binary arithmatic operations between two sets of images and or constants. The input can be any type of image, but the output is always a structure array. This program work on all images in memory and therefore require sufficient memory.
% sim_julday.m :  Get or calculate the Julian day of SIM images, based on their header information. This function is looking for the time and date in a set of header keywords and convert it to JD. The program can also use the exposure time to calculate the mid exposure time.
% sim_mask_saturated.m :  For each image in a list of images, look for saturated pixels and generate a bit mask image with the saturated pixels marked.
% sim_maskstars.m :  mask stars or high level pixels in a set of structure array images.
% sim_mosaic.m :  Mosaicing (tiling) a set of images into a single image.
% sim_reduce_set.m :  Reduce a set of images taken in the same configuration (e.g., identical image size, filter, Gain and RN). The reduction may include: flagging of saturated pixels, bias subtraction, flat field correction, CR flagging and removal.
% sim_replace.m :  Replace pixels in a give value ranges with other values in a set of structure images (SIM).
% sim_resize.m :  Resize a set of structure array of images using the imresize.m function. 
% sim_rotate.m :  Rotate a set of structure images (SIM).
% sim_scaleflux.m :  Scale the flux of SIM images.
% sim_shift.m :  Shift in X/Y coordinates a set of structure images (SIM). 
% sim_stat.m :  Given a set of images, calculate statistics of each image.
% sim_std.m :  Calculate the StD (or error) per image or per pixel for a set of SIM images.
% sim_suboverscan.m :  Calculate and subtract overscan bias from a list of images.
% sim_transform.m :  Aplay a spatial transformation to a list of SIM images.
% sim_trim.m :  Trim a set of images and save the result in a structure array. If needed, then additional associated images (e.g., mask) will be trimmed too.
% sim_ufun.m :  Operate a unary function on a set of structure images (SIM).
% sim_ufunv.m :  Operate a unary function that operate on all the elements of an image and return a scalar (e.g., @mean). The function are applied to a set of structure images (SIM).
% sim_update_head.m :  
% sim_xcorr.m :  Cross correlate two sets of images. The input can be any type of image, but the output is always a structure array. This program work on all images in memory and therefore require sufficient memory.
% sim_zeropad.m :  Pad SIM images with zeros.
% simcat2array.m :  Given a single elemnt structure array or SIM with 'Cat' and 'Col' fields, return a matrix containing the catalog fields, and the indices of some specified columns.
% simdef.m :  Define a SIM class with multiple entries (array). 
% sims2fits.m :  Write all the images stored in a structure image array (SIM) as a FITS files. See sim2fits.m for writing a single files.
% struct2sim.m :  Convert a structure array to SIM object.
% sum_prod_fft.m :  Calculate the sum of products of 2-D fast fourier transform of two cubes. I.e., calculate sum_{i}{conj(fft(P_i))*fft(R_i)/Sigma^2} were the conj is optional and sigma is an optional normalization. P_i and R_i are 2D arrays.
% weights4coadd.m :  Calculate image weights for optimal coaddition under various assumptions. Also return imaages statistics including the images Std, Variance, Background, RN, Gain, ZP, relative Transperancy, PSF, SigmaX, SigmaY, Rho and FWHM.
% xcat2sim.m :  cross-correlate external astronomical catalogs with a catalog or SIM catalog.
