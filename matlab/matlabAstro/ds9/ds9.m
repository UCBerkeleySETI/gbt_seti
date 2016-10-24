function ds9(varargin)
%--------------------------------------------------------------------------
% ds9 function                                                         ds9
% Description: Load images (FITS file, image or matlab matrix) into ds9.
%              The user should open ds9 before execution of this function.
%              THIS FUNCTION IS IDENTICAL TO ds_disp.m (a shortcut).
% Input  : - Image name - one of the followings:
%            A string containing a FITS file name to load, or a string
%            containing wild cards (see create_list.m for options).
%            A string that start with "@" in this case the string will
%            be regarded as a file name containing a list of images.
%            A matrix containing an image.
%            A cell array of FITS file names.
%            A cell array of matrices.
%            If empty, then only change the properties of the chosen frame.
%          - Frame - This can be a function {'new'|'first'|'last'|'next'|'prev'}
%            or a number, default is 'new'.
%            If the image name is a cell array then
%            the frame number can be a vector, otherwise it will
%            load the images to the sucessive frames.
%            Default is next frame.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            Keywords and values can be one of the followings:
%            'Scale'       - Scale function {'linear'|'log'|'pow'|'sqrt'
%                                            'squared'|'histequ'}
%                            default is 'linear'.
%            'ScaleLimits' - Scale limits [Low High], no default.
%            'ScaleMode'   - Scale mode {percentile |'minmax'|'zscale'|'zmax'},
%                            default is 'zscale'.
%            'CMap'        - Color map {'Grey'|'Red'|'Green'|'Blue'|'A'|'B'|
%                                       'BB'|'HE'|'I8'|'AIPS0'|'SLS'|'HSV'|
%                                       'Heat'|'Cool'|'Rainbow'|'Standard'|
%                                       'Staircase'|Color'},
%                            default is 'Grey'.
%             'InvertCM'   - Invert colormap {'yes'|'no'}, default is 'no'.
%             'Rotate'     - Image rotation [deg], default is to 0 deg.
%             'Orient'     - Image orientation (flip) {'none'|'x'|'y'|'xy'},
%                            default is 'none'.
%                            Note that Orient is applayed after rotation.
%             'Zoom'       - Zoom to 'fit' or [XY] or [X Y] zoom values,
%                            default is [2] (i.e., [2 2]).
%             'Pan'        - Center image on coordinates [X,Y],
%                            default is [] (do nothing).
%             'PanCoo'     - Coordinate system for Pan
%                            {'fk4'|'fk5'|'icrs'|'iamge'|'physical'},
%                            default is 'image'.
%             'Match'      - If more than one image is loaded, then
%                            match (by coordinates) all the images to
%                            the image in the first frame.
%                            Options are
%                            {'wcs'|'physical'|'image'|'none'},
%                            default is 'none'.
%             'MatchCB'    - If more than one image is loaded, then
%                            match (by colorbars) all the images to
%                            the image in the first frame.
%                            Options are {'none'|'colorbars'}
%                            default is 'none'.
%             'MatchS'     - If more than one image is loaded, then
%                            match (by scales) all the images to
%                            the image in the first frame.
%                            Options are {'none'|'scales'}
%                            default is 'none'.
%             'Tile'       - Tile display mode {'none'|'column'|'row'} or
%                            the number of [row col], default is 'none'.
% Output : null
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://wiezmann.ac.il/home/eofek/matlab/
% Example: ds9 20050422051427p.fits
% Reliable: 1
%-------------------------------------------------------------------------
ds9_disp(varargin{:});
