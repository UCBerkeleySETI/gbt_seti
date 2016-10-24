%
% Contents file for package: Swift
% Created: 29-Dec-2015
%---------
% build_chandra_obsid_cat.m :  Construct a catalog of all Chandra observations by going over the entire Chandra image archive.
% chandra_psf.m :  Read and interpolate the Chandra ACIS-S/ACIS-I PSF from the Chandra CalDB files. Instellation: 1. Install the Chandra CalDB directory including the 2d PSFs. 2. Modify the DefV.CalDBdir to direct to the Chandra CalDB directory.
% filter_badtimes.m :  Given a structure array containing X-ray event catalogs and a list of bad times, remove the bad times from the list.
% read_chandra_acis.m :  Read Chandra ACIS event files associated with a Chandra ObsID and add columns containing the RA and Dec for each event. If needed cd to osbid directory.
% search_filetemplate.m :  Given a directory name, search recursively within the directory and all its subdirectories, file names with a specific template. File path that contains a specific string can be excluded
% swxrt_coo2xy.m :  Given a Swift XRT event file (image) convert J2000.0 equatorial coordinates to X/Y (image) position. see also: swxrt_xy2coo.m
% swxrt_filter_badtimes.m :  Given an X-ray event file, look for time ranges in which the background is elevated above the mean background rate by a given amount.
% swxrt_getobs.m :  Look for Swift/XRT observations taken in a given coordinates. Return a link to the directories containing the observations, and retrieve the cleaned "pc" event file for each observation. Instelation: (1) install "browse_extract_wget.pl" from 	 http://heasarc.gsfc.nasa.gov/W3Browse/w3batchinfo.html (2) Set the RelPath variable in this program to specify the path of "browse_extract_wget.pl" script relative to the path of this program.
% swxrt_src.m :  Given a Swift/XRT image and a position. Measure the counts in the position and the bacground counts in an annulus around the position.
% swxrt_xy2coo.m :  Given a Swift XRT event file (image) convert X/Y (image) position to J2000.0 equatorial coordinates. see also: swxrt_coo2xy.m
% table_select.m :  Given a table (e.g., X-ray events), select events based on selection criteria.
% wget_chandra_obsid.m :  Get all the files associated with a Chandra ObsID
