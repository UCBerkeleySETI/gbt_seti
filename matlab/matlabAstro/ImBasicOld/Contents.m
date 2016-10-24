%
% Contents file for package: ImBasicOld
% Created: 29-Dec-2015
%---------
% bias_fits.m :  Given a set of images with the same size, look for the bias images, create a super bias image and subtract it from all the non-bias images. This function is obsolote: Use ImBasic toolbox.
% find_size_fits.m :  Given a list of 2D FITS images return the image sizes as specified in the image headers (NAXIS1, NAXIS2 keywords). Optionally, also look for all the unique sizes of images among the list and return sublists of all the images with a given size.
% flat_fits.m :  Given a set of images with the same size, construct a flat-field image for each filter and divide all the images of a given filter by the corresponding normalized flat field image. The script can look for twilight or dome flat images or alternatively construct a super flat image (see superflat_fits.m for details). Note that superflat_fits.m can remove stars from the images prior to creation of the flat image. This function is obsolote: Use ImBasic toolbox.
% flatcreate_fits.m :  Create a flat field from a single filter set of images. See also flat_fits.m; flatset.fits.m; optimal_flatnorm_fits.m This function is obsolote: Use ImBasic toolbox.
% flatset_fits.m :  Given a set of images from which to construct a flat field. Correct another set of images of the same size. This is a simpleer version of flat_fits.m which works on a predefined lists. All the images in these lists should be taken using the same filter (see also flat_fits.m). This function is obsolote: Use ImBasic toolbox.
% get_ccdsec_fits.m :  Get an parse CCD section keyword value from a list of images.
% identify_bias_fits.m :  Given a set of images, look for all the bias images. This script can be used also to identify errornous bias images.
% identify_flat_fits.m :  Given a set of images, look for all the flat images. The program also check if some of the flat images are saturated and if so reject them from the list of flat images.
% imarith_fits.m :  Perform a binary arithmatic operations between two sets of images and or constants. The input can be either FITS images, matrices or structure arrays (see read2sim.m), while the output is always a file. This function load and work and save images one by one so it is not memory extensive. Use imarith_sim.m if you want to operate on all images in memory. Use of imarith_sim.m may save time in reading/writing FITS images, but requires a lot of memory.
% imcombine_fits.m :  Combine FITS images. This function use imcombine.m. This function is obsolote: Use ImBasic toolbox.
% imconv_fits.m :  Convolve a FITS image with a kernel.
% imfft_fits.m :  Calculate the 2D Fast Fourier Transform of FITS images.
% imflip_fits.m :  Flip FITS images in x or y direction.
% imfun_fits.m :  Applay a function to a set of images. The function, may be operating on multiple images at a time.
% imget_fits.m :  Load a list of FITS images or mat files into a cell array of images. This is mainly used as a utility program by some of the *_fits.m function.
% imifft_fits.m :  Calculate the 2D inverse Fast Fourier Transform of FITS images.
% imreplace_fits.m :  Search for pixels with value in a given range and replace them with another value.
% imresize_fits.m :  Resize a FITS image. This is dones by interpolation or convolution (using an interpolation kernel) so effectively this can be used also to convolve an image.
% imrotate_fits.m :  Rotate FITS images.
% imstat_fits.m :  Given a list of FITS images or images in matrix format calculate various statistics for each image.
% imsubback_fits.m :  Subtract background from a 2-D matrix or a FITS image using various nethods.
% imxcorr_fits.m :  Cross correlate two FITS images using FFT and find the optimal linear shift between the two images. Optionally, the program subtract the background from the images before the cross-correlation step.
% keyword_grouping1_fits.m :  Group images by a single image header keyword. For example, can be used to select all images taken with a given filter. see also: keyword_grouping_fits.m
% keyword_grouping_fits.m :  Group images by multiple image header keywords. For example, can be used to select all images taken with a given filter and type. see also: keyword_grouping1_fits.m
% maskstars_fits.m :  Given a list of FITS images look for stars in these images or pixels with high value above background and replace these pixels by NaN or another value.
% split_multiext_fits.m :  Given a list of multi extension FITS files break each file to its multiple extensions. Each extension will be given the header of the primary FITS file with the extension header.
% trim_fits.m :  Trim a list of FITS images or 2D arrays.
