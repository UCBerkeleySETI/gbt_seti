function OutTemp=convert_temp(InTemp,InUnits,OutUnits)
%--------------------------------------------------------------------------
% convert_temp function                                            General
% Description: Convert between temperature systems.
% Input  : - Input temperature.
%          - Units of the input temperature:
%            'C' - degree Celsius
%            'F' - degree Fahrenheit
%            'K' - Kelvin
%            'R' - Rankine
%          - Units of output temperature.
% Output : - Output temperature.
% Tested : Matlab 7.6
%     By : Eran O. Ofek                    Oct 2008
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutTemp=convert_temp(0,'C','K');
% Reliable: 2
%--------------------------------------------------------------------------


switch InUnits
 case 'K'
    TempK = InTemp;
 case 'C'
    TempK = InTemp + 273.15;
 case 'F'
    TempK = (InTemp + 459.67).*5./9;
 case 'R'
    TempK = InTemp.*5./9;
 otherwise
    error('Unknown temperature units');
end

% from Kelvin to OutUnits

switch OutUnits
 case 'K'
    OutTemp = TempK;
 case 'C'
    OutTemp = TempK - 273.15;
 case 'F'
    OutTemp = TempK.*9./5 - 459.67;
 case 'R'
    OutTemp = TempK.*9./5;
 otherwise
    error('Unknown temperature units');
end
