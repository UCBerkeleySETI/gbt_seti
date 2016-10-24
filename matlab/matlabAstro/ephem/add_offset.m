function [OutRA,OutDec]=add_offset(RA,Dec,Offset,PA)
%--------------------------------------------------------------------------
% add_offset function                                                ephem
% Description: Add an offset specified by angular distance and position
%              angle to celestial coordinates.
% Input  : - RA [radians].
%          - Dec [radians].
%          - Offset (angular distance) [radians].
%          - Position angle [radians].
% Output : - RA [radians].
%          - Dec [radians].
% See also: sphere_offset.m; sphere_dist.m
% Tested : Matlab 7.8
%     By : Eran O. Ofek                    Mar 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 1
%--------------------------------------------------------------------------

[OutDec,OutRA]=reckon(Dec,RA,Offset,PA,'radians');
