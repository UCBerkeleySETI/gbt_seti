function Date=jd2date(JD,Format);
%------------------------------------------------------------------------------
% jd2date function                                                       ephem
% Description: Convert Julian days to Gregorian/Julian date.
% Input  : - Row vector of (positive) Julian Days.
%          - Output format:
%            'f'  - [Day Month Year, Day_Fraction(UT)] (default).
%            'H'  - [Day Month Year, H M S]
% Output : - Matrix of dates.
%            e.g., [Day, Month, Year, Day_Fraction(UT)].
% Tested : Matlab 5.2
%     By : Eran O. Ofek                 September 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% See also: julday.m, julday1.m, date_axis.m
% Example: jd2date(julday([1 1 2000; 2 2 2000]).')
% Reliable: 1
%------------------------------------------------------------------------------
if (min(JD)<0),
   error('The method is valid only for poitive JDs');
end

if (nargin==1),
   Format = 'f';
end

Z = floor(JD+0.5);
F = (JD+0.5) - floor(JD+0.5);

A     = zeros(size(JD));
Day   = zeros(size(JD));
Month = zeros(size(JD));
Year  = zeros(size(JD));

IZ_s = find(Z<2299161);
IZ_l = find(Z>=2299161);
A(IZ_s) = Z(IZ_s);

Alpha = fix((Z(IZ_l) - 1867216.25)./36524.25);
A(IZ_l) = Z(IZ_l) + 1 + Alpha - fix(Alpha./4);

B = A + 1524;
C = fix((B - 122.1)./365.25);
D = fix(365.25.*C);
E = fix((B-D)./30.6001);

Day   = B - D - fix(30.6001.*E) + F;
IE_s  = find(E<14);
IE_l  = find(E>=14);
Month(IE_s) = E(IE_s) - 1;
Month(IE_l) = E(IE_l) - 13;

IM_l  = find(Month>2);
IM_s  = find(Month==1 | Month==2);
Year(IM_l) = C(IM_l) - 4716;
Year(IM_s) = C(IM_s) - 4715;
 
Date = [floor(Day), Month, Year, F];

switch lower(Format)
 case 'f'
    % do nothing
 case 'h'
    Date = [Date(:,1:3), convertdms(Date(:,4),'f','H')];
 otherwise
    error('unknown Format option');
end
