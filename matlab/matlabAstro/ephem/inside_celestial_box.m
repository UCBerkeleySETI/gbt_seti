function I=inside_celestial_box(RA,Dec,BoxRA,BoxDec,Width,Height);
%------------------------------------------------------------------------------
% inside_celestial_box function                                          ephem
% Description: Given a list of celestial coordinates, and a box center,
%              width and height, where the box sides are parallel to the
%              coorinate system, 
%              check if the coordinates are within the box.
% Input  : - Vector of celestial longitudes [radians] to test.
%          - Vector of celestial latitudes [radians] to test.
%          - Vector of longitudes of center of boxes [radians].
%          - Vector of latitudes of center of boxes [radians].
%          - Vector of full widths of boxes [radians].
%          - Vector of full heights of boxes [radians].
% Output : - Flag indicating if coordinate is inside corresponding box
%            (1) or not (0).
% Tested : Matlab 7.8
%     By : Eran O. Ofek                     March 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------

RAD = 180./pi;

[OffsetLong,OffsetLat,Dist,PA]=sphere_offset(BoxRA,BoxDec,RA,Dec,'rad','dr');

I = (abs(OffsetLat)<=(0.5.*Height) & abs(OffsetLong)<=(0.5.*Width));

