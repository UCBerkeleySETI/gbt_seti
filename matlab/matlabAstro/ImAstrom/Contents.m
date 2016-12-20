%
% Contents file for package: ImAstrom
% Created: 29-Dec-2015
%---------
% coo_trans.m :  Coordinates transformation given a general transformation.
% fit_affine2d_ns.m :  Fit an 2D affine transformation, non-simultanously to both axes, to a set of control points. The fit is of the form: Xref = A*X - B*Y + C Yref = D*X + E*Y + F
% fit_affine2d_s.m :  Fit an 2D affine transformation, simultanously to both axes, to a set of control points. The fit is of the form: Xref = A*X - B*Y + C Yref = A*Y + B*X + D
% fit_general2d_ns.m :  Fit (or apply) a 2D general transformation, non-simultanously to both axes, to a set of control points. The fit is of the form: Xref = A + B*X - C*Y + D*AM*sin(Q1) + E*AM*sin(Q2) + ... F*AM*Color*sin(Q1) + G*AM*Color*sin(Q1) + ... PolyX(X) +PolyX(Y) + PolyX(R) + H*FunX(pars) Yref = A + B*X - C*Y + D*cos(Q1) + E*cos(Q2) + ... F*AM*Color*cos(Q1) + G*AM*Color*cos(Q1) + ... PolyY(X) +PolyY(Y) + PolyY(R) + H*FunY(pars) If the free parameters are supplied than the Cat input is transformed to the Ref system.
% fit_tran2d.m :  Fit a general transformation between two sets of control points (catalog and reference).
% fits_get_wcs.m :  Get the WCS keywords information from a SIM image structure array or FITS images.
% get_fits_wcs.m :  Get WCS keywords from FITS image. Obsolote - See instead fits_get_wcs.m
% match_lists_shift.m :  Given two lists containing two dimensional planar coordinates (X, Y), find the possible shift transformations between the two lists, using a histogram of all possible combination of X and Y distances. 
% read_ctype.m :  Given a FITS header CTYPE keyword value (e.g., 'RA---TAN') return the coordinate type (e.g., RA) and transformation type (e.g., 'TAN').
% sim_align_shift.m :  Given a set of images and a reference image, register (align) the images to the reference image. This function is suitable for transformation which are mostly shifts (with some small rotation and distortions).
% sim_footprint.m :  Return the footprint (verteces of images footprint) and center for a list of images.
% sim_getcoo.m :  Get J2000.0 R.A. and Dec. from image header or SIM header.
% sky2xy.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), convert longitude and latitude to X and Y position in the image.
% sky2xy_ait.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), where the WCS is represented using the Hammer-Aitoff projection, convert longitude and latitude to X and Y position.
% sky2xy_tan.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), where the WCS is represented using the tangential projection, convert longitude and latitude position to X and Y in the image.
% swarp.m :  Run SWarp.
% tri_match_lsq.m :  Given two matrices of [X,Y, Mag] columns, where the Mag column is optional, attempt to find a shift+scale+rotation transformation between the two lists using triangle pattren matching.
% tri_match_lsq2.m :  Given two matrices of [X,Y, Mag] columns, where the Mag column is optional, attempt to find a shift+scale+rotation transformation between the two lists using triangle pattren matching.
% xy2sky.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), convert X and Y position in the image to longitude and latitude.
% xy2sky_ait.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), where the WCS is represented using the Hammer-Aitoff projection, convert X and Y position in the image to longitude and latitude.
% xy2sky_tan.m :  Given a FITS image, SIM or a structure containing the FITS WCS keyword (returned by fits_get_wcs.m), where the WCS is represented using the tangential projection, convert X and Y position in the image to longitude and latitude.
