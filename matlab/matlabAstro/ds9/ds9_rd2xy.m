function [CooX,CooY]=ds9_rd2xy(RA,Dec,CooType)
%------------------------------------------------------------------------------
% ds9_rd2xy function                                                       ds9
% Description: Convert J2000.0 RA/Dec in current ds9 display to physical
%              or image X/Y coordinates.
% Input  : - RA [rad] or [H M S] or sexagesimal string.
%          - Dec [rad] or [Sign D M S] or sexagesimal string.
%          - Type of output coordinates: {'image'|'physical'},
%            default is 'image'.
% Output : - Vector of X coordinates.
%          - Vector of Y coordinates.
% See also: fits_xy2rd.m, fits_rd2xy.m
% Tested : Matlab 7.0
%     By : Eran O. Ofek                     April 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [X,Y]=ds9_rd2xy([1 36 50],[1 15 48 11],'physical');
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;

DefN       = 1;
DefCooType = 'image';
if (nargin==2),
   CooType = DefCooType;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end



RA  = convertdms(RA,'gH','d');
Dec = convertdms(Dec,'gD','d');

N = length(RA);


CooX  = zeros(N,1);
CooY  = zeros(N,1);
Value = zeros(N,1);


for I=1:1:N,
   %--- set coordinates of crosshair ---
   ds9_system(sprintf('xpaset -p ds9 crosshair %f %f wcs icrs',RA(I),Dec(I)));
   %--- get Coordinates of crosshair ---
   [Status,CooIm] = ds9_system(sprintf('xpaget ds9 crosshair %s',CooType));
   %CooIm
   [CooX(I), CooY(I)] = strread(CooIm,'%f %f',1); %,'headerlines',4);
end


ds9_system(sprintf('xpaset -p ds9 mode pointer'));
