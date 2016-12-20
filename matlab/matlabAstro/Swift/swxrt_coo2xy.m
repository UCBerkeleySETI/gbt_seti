function [X,Y]=swxrt_coo2xy(File,RA,Dec,Tel);
%------------------------------------------------------------------------------
% swxrt_coo2xy function                                                  Swift
% Description: Given a Swift XRT event file (image) convert J2000.0
%              equatorial coordinates to X/Y (image) position.
%              see also: swxrt_xy2coo.m
% Input  : - FITS binary table of events file name.
%          - J2000.0 R.A. [radians].
%          - J2000.0 Dec. [radians].
%          - {'swift' | 'chandra'}. Default is 'swift'.
% Output : - X coordinates.
%          - Y coordinates.
% Tested : Matlab 7.13
%     By : Eran O. Ofek                  November 2011
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 1
%------------------------------------------------------------------------------

Def.Tel = 'swift';
if (nargin==3),
   Tel = Def.Tel;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of inputa rguments');
end

RAD = 180./pi;
switch lower(Tel)
 case 'swift'
    Keys       = {'TCRPX2','TCRVL2','TCDLT2','TCRPX3','TCRVL3','TCDLT3'};
 case 'chandra'
    Keys       = {'TCRPX11','TCRVL11','TCDLT11','TCRPX12','TCRVL12','TCDLT12'};
 case 'nustar'
    Keys       = {'TCRPX37','TCRVL37','TCDLT37','TCRPX38','TCRVL38','TCDLT38'};
 otherwise
    error('Unsupported telescope option');
end
%Keys       = {'TCRPX2','TCRVL2','TCDLT2','TCRPX3','TCRVL3','TCDLT3'};
[KeywordVal,KS] = get_fits_keyword(File,Keys,1,'BinaryTable');
KS.TCRPX2  = KeywordVal{1};
KS.TCRVL2  = KeywordVal{2};
KS.TCDLT2  = KeywordVal{3};
KS.TCRPX3  = KeywordVal{4};
KS.TCRVL3  = KeywordVal{5};
KS.TCDLT3  = KeywordVal{6};


[X,Y]=pr_gnomonic(RA,Dec,1,[KS.TCRVL2, KS.TCRVL3]./RAD);
X = X.*RAD./KS.TCDLT2 + KS.TCRPX2;
Y = Y.*RAD./KS.TCDLT3 + KS.TCRPX3;

