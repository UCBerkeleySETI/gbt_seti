function Area=area_sphere_polygon(PolyLon,PolyLat);
%------------------------------------------------------------------------------
% area_sphere_polygon function                                           ephem
% Description: Calculate the area of a polygon on a sphere, where the
%              polygon sides are assumed to be great circles. If the polygon
%              is not closed (i.e., the first point is identical to the last
%              point) then, the program close the polygon.
% Input  : - Column vector of longitudes for polygon.
%            Units are [rad] or [H M S] or sexagesimal.
%          - Column vector of latitude for polygon.
%            Units are [rad] or [Sign D M S] or sexagesimal.
% Output : - Area of spherical polygon which its sides are great circles.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                     March 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------
  error('corners angles not debuged');

RAD = 180./pi;
Threshold = 0.00001./(RAD.*3600);    % threshold for definition for identical points

PolyLon = convertdms(PolyLon,'gH','r');
PolyLat = convertdms(PolyLat,'gD','R');

[Dist,PA] = sphere_dist(PolyLon(1),PolyLat(1),PolyLon(end),PolyLat(end));
if (Dist<Threshold),
   % polygon is closed
else
   % close polygon
   PolyLon = [PolyLon; PolyLon(1)];
   PolyLat = [PolyLat; PolyLat(1)];
end
% add last
PolyLon = [PolyLon(end-1);PolyLon];
PolyLat = [PolyLat(end-1);PolyLat];

N = length(PolyLon) - 2;

SumAng = 0;
for I=2:1:N+1,
   [Dist,PA1] = sphere_dist(PolyLon(I-1),PolyLat(I-1),PolyLon(I+1),PolyLat(I+1));
   [Dist,PA2] = sphere_dist(PolyLon(I),PolyLat(I),PolyLon(I+1),PolyLat(I+1));
   SumAng = SumAng + (PA2-PA1);
end

Area = abs(SumAng - (N-2).*pi);
