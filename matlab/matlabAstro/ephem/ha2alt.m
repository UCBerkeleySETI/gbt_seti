function [Alt,AM]=ha2alt(HA,Dec,Lat)
%--------------------------------------------------------------------------
% ha2alt function                                                    ephem
% Description: Given Hour Angle as measured from the meridian, the source
%              declination and the observer Geodetic latitude, calculate
%              the source altitude above the horizon and its airmass.
% Input  : - Hour Angle [radians].
%          - Declination [radians].
%          - Latitude [radians].
% Output : - Altitude [radians].
%          - Airmass.
% See also: horiz_coo.m, ha2az.m
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Aug 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 1
%--------------------------------------------------------------------------

Alt = asin(sin(Dec).*sin(Lat) + cos(Dec).*cos(Lat).*cos(HA));
AM  = hardie(pi./2-Alt);
