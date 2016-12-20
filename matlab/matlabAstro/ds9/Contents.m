%
% Contents file for package: ds9
% Created: 29-Dec-2015
%---------
% ds9.m :  Load images (FITS file, image or matlab matrix) into ds9. The user should open ds9 before execution of this function. THIS FUNCTION IS IDENTICAL TO ds_disp.m (a shortcut).
% ds9_disp.m :  Load images (FITS file, image or matlab matrix) into ds9. The user should open ds9 before execution of this function.
% ds9_dispsex.m :  Display markers around a list of sources on an image displayed using ds9.
% ds9_exam.m :  Interactive examination of an image displayed in ds9. The program display an image in ds9 and then prompt the user to use various clicks to examin sources, vectors, and regions.
% ds9_frame.m :  Switch ds9 display to a given frame number.
% ds9_get_filename.m :  Get the file name of the image which is displayed in the current ds9 frame.
% ds9_getbox.m :  Get from the ds9 display the pixel values in a specified box region.
% ds9_getcoo.m :  Interactively get the coordinates (X/Y or WCS) and value of the pixel selected by the mouse (left click) or by clicking any character on the ds9 display.
% ds9_getvecprof.m :  Given the X and Y coordinates of two points, and an open ds9 display, get the value of the image in the display, interpolated along the line connecting the two points.
% ds9_imserver.m :  load an image from one of the ds9 image servers into the ds9 display.
% ds9_lineprof.m :  Interactive examination of vector in a FITS image displayed in ds9. The program display an image in ds9 and then prompt the user to click on two points in the image. Than the program reads from the image a vector between the two points and return/display the vector. THIS FUNCTION WILL BE REMOVED - USE ds9_exam.m
% ds9_loop_disp.m :  Load a list of images (FITS file) into ds9 one by one. Prompt the user for the next image. Open ds9 before execution of this function.
% ds9_phot.m :  Interactive photometry for ds9 display. Allow the user to mark objects on the ds9 display, and perform centeroiding, and simple aperture photometry.
% ds9_plotregion.m :  Write a region file containing various plots and load it to the ds9 display.
% ds9_plottrace.m :  Given a list of X and Y plot Y(X) on the ds9 display.
% ds9_print.m :  Print the current frame to PostScript or JPG file.
% ds9_rd2xy.m :  Convert J2000.0 RA/Dec in current ds9 display to physical or image X/Y coordinates.
% ds9_regions.m :  Load, save or delete ds9 regions file.
% ds9_save.m :  Save an image in the ds9 dispaly as a FITS image.
% ds9_sdssnavi.m :  Click on a position in an image displayed in ds9 and this program will open the SDSS navigator web page for the coordinates.
% ds9_slit.m :  Download an image from a ds9 server and plot a slit in the ds9 display. profile in the image along the slit.
% ds9_start.m :  Checks whether ds9 is running, and if it is not running then start ds9.
% ds9_system.m :  Matlab sets DYLD_LIBRARY_PATH incorrectly on OSX for ds9 to work. This function is a workround
% ds9simcat.m :  Display SIM images (or any other images) in ds9, create or use associate source catalog, query the source catalog and display markers around selected sources.
