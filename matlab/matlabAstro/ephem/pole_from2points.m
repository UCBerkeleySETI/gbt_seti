function [RA3,Dec3]=pole_from2points(RA1,Dec1,RA2,Dec2);
%------------------------------------------------------------------------------
% pole_from2points function                                              ephem
% Description: Given two points on the celestial sphere (in any system)
%              describing the equator of a coordinate system,
%              find one of the poles of this coordinate system.
% Input  : - RA of 1st point [radians].
%          - Dec of the 1st point [radians]
%          - RA of the 2nd point [radians]
%          - Dec of the 2nd point [radians]
% Output : - RA of one of the poles.
%          - Dec of the pole.
% Tested : Matlab 7.3
%     By : Eran O. Ofek                      July 2008
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------

CosD  = sin(Dec1).*sin(Dec2) + cos(Dec1).*cos(Dec2).*cos(RA2-RA1);
D     = acos(CosD);
Gamma = asin(cos(Dec2).*sin(RA2-RA1)./sin(D)) - pi./2;
Dec3  = asin(cos(Dec1).*cos(Gamma));
RA3   = RA1 + atan2(sin(Gamma)./cos(Dec3),-sin(Dec1).*cos(Gamma)./cos(Dec3));
