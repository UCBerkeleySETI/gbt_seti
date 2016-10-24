function [Ind]=find_ranges(Vector,Ranges)
%------------------------------------------------------------------------------
% find_ranges function                                                 General
% Description: Given a vector and several ranges, return the indices of
%              values in the vector which are found within one of the ranges.
% Input  : - Column vector of values.
%          - Matrix of ranges, in which each row specify a range.
%            The first column is for the range lower bound and the
%            second colum for the upper bound.
% Output : - List of indices.
% Tested : Matlab 7.3
%     By : Eran O. Ofek                       Feb 2008
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 1
%--------------------------------------------------------------------------

Nr = size(Ranges,1);
Ind = [];
for Ir=1:1:Nr,
   Ind = [Ind; find(Vector>=Ranges(Ir,1) & Vector<=Ranges(Ir,2))];
end

