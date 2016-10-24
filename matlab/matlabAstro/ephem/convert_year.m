function Output=convert_year(Input,InType,OutType)
%--------------------------------------------------------------------------
% convert_year function                                              ephem
% Description: Convert between different types of years. For example, this
%              program can convert Julian years to Besselian years or JD
%              and visa versa.
% Input  : - Input to convert;
%          - Input type - options are:
%            'J'    - Julian year.
%            'B'    - Besselian year.
%            'JD'   - Julian days.
%            'MJD'  - MJD.
%            'Date' - date matrix [D M Y] or [D M Y frac] or [D M Y H M S].
%          - Output type, options are as the input type.
% Output : - Output.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Oct 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: convert_year(2000,'J','B')
% Reliable: 2
%--------------------------------------------------------------------------

switch lower(InType)
 case 'j'
    % convert Julian years to JD
    JD = (Input - 2000).*365.25 + 2451545.0;
 case 'b'
    % convert Besselian years to JD
    JD = (Input - 1900).*365.2421988 + 2415020.3135;
 case 'jd'
    JD = Input;
 case 'mjd'
    JD = Input + 2400000.5;
 otherwise
    error('Unknown InType option');
end


switch lower(OutType)
 case 'j'
    % convert JD to Julian years
    Output = 2000 + (JD-2451545.0)./365.25;
 case 'b'
    % convert JD to Besselian years
    Output = 1900 + (JD-2415020.3135)./365.2421988;
 case 'jd'
    Output = JD
 case 'mjd'
    Output = JD - 2400000.5;
 otherwise
    error('Unknown OutType option');
end



 
