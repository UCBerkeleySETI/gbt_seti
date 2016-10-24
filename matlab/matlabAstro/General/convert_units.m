function [Factor,Out,String]=convert_units(InUnits,OutUnits,In)
%--------------------------------------------------------------------------
% convert_units function                                           General
% Description: Unit conversion function. Given an input and output strings
%              containing unit names, return the conversion multiplication 
%              factor needed for converting the input units to the output
%              units. The user is responsible for the balance of the
%              transformation.
%              Type of units:
%               Length:
%                  'mm' ; 'cm' ; 'inch' ; 'feet' ;
%                  'm' - meter ; 'km' ; 'mile' ;
%                  'erad' - Earth radius ; 'au' ;
%                  'ly' - light year ; 'pc'; 'yard'
%                Time:
%                  's' ; 'min' ; 'hour' ; 'day';
%                  'sday' - sidereal day ; week ;
%                  'year'; 'cen' - century
%                Mass:
%                  'gr'; 'kg'; 'emass' - Earth mass;
%                  'jmass' - Jupiter mass;
%                  'smass' - Solar mass;
%                  'mp' - proton mass;
%                  'me' - electron mass;
%                  'libra';'pound'
%                Energy: (see also convert_energy.m)
%                  'erg'; 'J'
%                Angle:
%                  'rad' ; 'deg' ;
%                  'amin' (or 'arcmin') - arcmin ; 'asec' (or 'arcsec') - arcsec
%                Solid Angle:
%                  'ster' ; 'sdeg' - square degree ;
%                  'smin' - square arcmin ;
%                  'ssec' - square arcsec
% Input  : - String containing the input units.
%          - String containing the output units.
%          - Optional value to convert from input units to output
%            units.
% Output : - Multiplication factor for converting input units to
%            output units.
%          - The input value given in the output units.
%          - A string describing the transformation (e.g., 
% Example : convert_units('m^3 * kg^-1 * s^-2','cm^3 * gr^-1 * s^-2')
% Tested : Matlab 6.5
%     By : Eran O. Ofek                    Jul 2003
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Factor,Out,String]=convert_units('erg','J',1)
% Reliable: 1
%--------------------------------------------------------------------------
RAD  = 180./pi;

%--- Length units ---
mm   = 1e-3;                         % mm
cm   = 1e-2;                         % cm
inch = 0.0254;                       % inch
feet = 0.30480;                      % foot
yard = 0.9144;                       % yard
m    = 1;                            % meter
km   = 1000;                         % km
mile = 1609;                         % mile
erad = refellipsoid('WGS84');
erad = erad(1);                      % Earth equatorial radius (WGS84)
au   = get_constant('au','SI');      % au
ly   = get_constant('ly','SI');      % light-year
pc   = get_constant('pc','SI');      % pc


%--- Time units ---
s    = 1;                            % sec
min  = 60;                           % minute
hour = 3600;                         % hour
sday = 86164.09053;                  % sidereal day
day  = 86400;                        % day
week = 7.*day;                       % week
year = 365.25.*day;                  % tropical year
cen  = year*100;                     % tropical century

%--- Mass units ---
gr   = 1e-3;                         % gram
kg   = 1;                            % kg
emass= get_constant('EarM','SI');    % Earth Mass
jmass= get_constant('JupM','SI');    % Jupiter Mass
smass= get_constant('SolM','SI');    % Solar Mass
me   = get_constant('me','SI');      % electron mass
mp   = get_constant('mp','SI');      % proton mass
libra= 0.32736409;                   % libra
pound= 0.45359237;                   % pound

%--- Energy Units ---
erg  = 1e-7;                         % erg
J    = 1;                            % joul

%--- Angle units ---
rad  = 1;                            % radian
deg  = 1./RAD;                       % degree
amin = deg./60;                      % arcmin
asec = amin./60;                     % arcsec
arcmin = deg./60;
arcsec = arcmin./60;

%--- Solid Angle units ---
ster = 1;                            % steradian
sdeg = 1./(RAD.^2);                  % square degree
smin = 1./((60.*RAD).^2);            % square arcmin
ssec = 1./((3600.*RAD).^2);          % square arcsec


%--- Find conversion factor ---
F1   = eval(InUnits);
F2   = eval(OutUnits);

Factor = F1./F2;

if (nargin>2),
   Out = Factor.*In;
end

if (nargout>2),
   String = sprintf('1 %s = %g %s',InUnits,Factor,OutUnits);
end

