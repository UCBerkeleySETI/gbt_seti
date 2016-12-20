%
% Contents file for package: GALEX
% Created: 29-Dec-2015
%---------
% coo2galexid.m :  Given a celestial equatorial coordinates, find all GALEX Vsn/Tilenum/Type/Ow/Prod/Img/Try number ID that cover the coordinates.
% get_galex_corrim.m :  Given an GALEX field ID [Vsn, Tilenum, Type, Ow, Prod, Img, Try], get the link to, and the GALEX intensity/rrhr or count image in FITS format. Furthermore, read the corrected image into matlab matrix. The program first check if the FITS image exists in current directory and if so, it reads the image from the disk. Note that if nargout>1 then the fits file is retrieved.
