function JD=year2jd(Year)
%--------------------------------------------------------------------------
% year2jd function                                                   ephem
% Description: Return the Julian day at Jan 1 st of a given list of years.
% Input  : - Column vector of years.
% Output : - Vector of JDs at Year Jan 1.0.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jun 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: JD=year2jd(2000);
% Reliable: 2
%--------------------------------------------------------------------------

Ny = numel(Year);
JD = julday([ones(Ny,2), Year]);

