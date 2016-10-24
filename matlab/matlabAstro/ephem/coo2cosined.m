function [CD1,CD2,CD3]=coo2cosined(Long,Lat)
%--------------------------------------------------------------------------
% coo2cosined function                                               ephem
% Description: Convert coordinates to cosine directions in the same
%              reference frame. See also: cosined.m, cosined2coo.m
% Input  : - Matrix of longitudes [radians].
%          - Matrix of latitudes [radians].
% Output : - Matrix of first cosine directions.
%          - Matrix of second cosine directions.
%          - Matrix of third cosine directions.
% Tested : Matlab 7.10
%     By :  Eran O. Ofek                   Oct 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [CD1,CD2,CD3]=coo2cosined(Long,Lat);
% Reliable: 1
%--------------------------------------------------------------------------

CosLat = cos(Lat);
CD1 = cos(Long).*CosLat;
CD2 = sin(Long).*CosLat;
CD3 =            sin(Lat);

%CD1 = cos(Long).*cos(Lat);
%CD2 = sin(Long).*cos(Lat);
%CD3 =            sin(Lat);
