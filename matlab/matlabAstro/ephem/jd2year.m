function Year=jd2year(JD,Type)
%--------------------------------------------------------------------------
% jd2year function                                                   epehm
% Description: Convert Julian day to Julian or Besselian years.
% Input  : - Vector of Julian days.
%          - Type of output years: 
%            'J'  - Julian year (default).
%            'B'  - Besselian year.
% Output : - Vector of (Julian or Besselian) years with decimal years.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Apr 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: year2jd.m
% Reliable: 1
%--------------------------------------------------------------------------
DefType  = 'J';
if (nargin==1),
   Type   = DefType;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end


switch lower(Type)
 case 'j'
    JulYear  = 365.25;
    JD0      = julday([1 1 2000 0]);
    Year     = 2000 + (JD - JD0)./JulYear;
 case 'b'
    BesYear  = 365.242189;
    JD0      = 2451544.53;
    Year     = 2000 + (JD - JD0)./BesYear;
 otherwise
    error('Unknown Type option');
end
