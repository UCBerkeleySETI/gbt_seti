function [Value,Units,Const,ConstS]=get_constant(ConstantName,System);
%------------------------------------------------------------------------------
% get_constant function                                                General
% Description: Get the value of an astronomical/physical constant.
% Input  : - Either a Constant name from the following list,
%            or a constant full name. The progarm first search the 
%            constant name and only if nothing found it is looking
%            for it using its full name.
%            'au'   - Astronomical Unit (DE405; Standish 1998)
%            'G'    - Gravitational constant
%            'c'    - Speed of light
%            'h'    - Planck constant
%            'e'    - elementry charge
%            'alpha'- fine structure constant
%            'Ryd'  - Rydberg constan 
%            'mp'   - Proton mass
%            'me'   - electron mass
%            'amu'  - Atomic mass unit
%            'NA'   - Avogadro constant
%            'kB'   - Boltzmann constant
%            'sigma'- Stefan-Boltzmann constant
%            'sigmaT'-Thomson Cross Section
%            'BohrR' - Bohr radius.
%            'SolM' - Solar mass
%            'EarM' - Earth mass
%            'EarR' - Earth mean radius
%            'JupM' - Jupiter mass
%            'JupR' - Jupiter mean radius
%            'SolL' - Solar luminosity
%            'SolR' - Solar radius
%            'pc'   - Parsec
%            'ly'   - light year (tropical)
%            'k'    - Gaussian gravitational constant (0.017...)
%          - Unit system:
%            'SI'  - meter-kg-sec
%            'cgs' - cm-gram-sec, default
% Output : - Constant value.
%          - Constant units.
%          - Structure array containing all constants.
%          - Structure containing all constants, where fields are constant
%            names.
% Tested : Matlab 6.5
%     By : Eran O. Ofek                    Jul 2003
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [c,Units]=get_constant('c','cgs');  % get the speed of light
% ToDo: http://physics.nist.gov/cuu/Constants/Table/allascii.txt
% Reliable: 2
%------------------------------------------------------------------------------
if (nargin==1),
   System = 'cgs';
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

I = 0;
I = I + 1;
Const(I).Name      = 'au';
Const(I).Desc      = 'astronomical unit';
Const(I).Form      = '';
Const(I).si.Val    = 1.49597870691e11;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 1.49597870691e13;
Const(I).cgs.Units = 'cm';
I = I + 1;
Const(I).Name      = 'G';
Const(I).Desc      = 'newton gravitational constant';
Const(I).si.Val    = 6.67259e-11;
Const(I).si.Units  = 'm^3 * kg^-1 * s^-2';
Const(I).cgs.Val   = 6.67259e-8;
Const(I).cgs.Units = 'cm^3 * gr^-1 * s^-2';
I = I + 1;
Const(I).Name      = 'c';
Const(I).Desc      = 'speed of light in vacum';
Const(I).Form      = '';
Const(I).si.Val    = 299792458;
Const(I).si.Units  = 'm * s^-1';
Const(I).cgs.Val   = 29979245800;
Const(I).cgs.Units = 'cm * s^-1';
I = I + 1;
Const(I).Name      = 'h';
Const(I).Desc      = 'planck constant';
Const(I).Form      = '';
Const(I).si.Val    = 6.6260755e-34;
Const(I).si.Units  = 'm^2 * kg * s^-1';
Const(I).cgs.Val   = 6.6260755e-27;
Const(I).cgs.Units = 'cm^2 * gr * s^-1';
I = I + 1;
Const(I).Name      = 'e';
Const(I).Desc      = 'elementary electric charge';
Const(I).Form      = '';
Const(I).si.Val    = 1.60217733e-19;
Const(I).si.Units  = 'C';
Const(I).cgs.Val   = 4.8032068e-10;
Const(I).cgs.Units = 'esu';
I = I + 1;
Const(I).Name      = 'alpha';
Const(I).Desc      = 'fine structure constant';
Const(I).Form      = '';
Const(I).si.Val    = 7.2973525698e-3;
Const(I).si.Units  = '';
Const(I).cgs.Val   = 7.2973525698e-3;
Const(I).cgs.Units = '';
I = I + 1;
Const(I).Name      = 'Ryd';
Const(I).Desc      = 'Rydberg';
Const(I).Form      = 'me * e^4/(8*eps0^2 * h^3 * c)';
Const(I).si.Val    = 10.973731568539e6;
Const(I).si.Units  = 'm^-1';
Const(I).cgs.Val   = 10.973731568539e4;
Const(I).cgs.Units = '';
I = I + 1;
Const(I).Name      = 'eps0';
Const(I).Desc      = 'vacuum permittivity/permittivity of free space';
Const(I).Form      = '1/(mu0 * c^2) = e^2/(2*alpha*h*c)';
Const(I).si.Val    = 8.854187817620e-12;
Const(I).si.Units  = 'F * m^-1';
Const(I).cgs.Val   = NaN;
Const(I).cgs.Units = '';
I = I + 1;
Const(I).Name      = 'mu0';
Const(I).Desc      = 'vacuum permeability/magnetic constant';
Const(I).Form      = '1/(eps0 * c^2)';
Const(I).si.Val    = 4.*pi.*1e-7;
Const(I).si.Units  = 'H * m^-1';
Const(I).cgs.Val   = NaN;
Const(I).cgs.Units = '';
I = I + 1;
Const(I).Name      = 'mp';
Const(I).Desc      = 'proton rest mass';
Const(I).Form      = '';
Const(I).si.Val    = 1.6726231e-27;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 1.6726231e-24;
Const(I).cgs.Units = 'gr';
I = I + 1;
Const(I).Name      = 'me';
Const(I).Desc      = 'electron rest mass';
Const(I).Form      = '';
Const(I).si.Val    = 9.10938997e-31;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 9.10938997e-28;
Const(I).cgs.Units = 'gr';
Const(I).error     = 4.9e-10;
I = I + 1;
Const(I).Name      = 'amu';
Const(I).Desc      = 'atomic mass unit';
Const(I).Form      = '';
Const(I).si.Val    = 1.660538921e-27;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 1.660538921e-24;
Const(I).cgs.Units = 'gr';
Const(I).error     = 4.4e-8;
I = I + 1;
Const(I).Name      = 'NA';
Const(I).Desc      = 'avogadro constant/avogadro number';
Const(I).Form      = '';
Const(I).si.Val    = 6.0221412927e23;
Const(I).si.Units  = 'mol^-1';
Const(I).cgs.Val   = 6.0221412927e23;
Const(I).cgs.Units = 'mol^-1';
Const(I).error     = 4.5e-10;
I = I + 1;
Const(I).Name      = 'kB';
Const(I).Desc      = 'boltzmann constant';
Const(I).Form      = 'R/NA';
Const(I).si.Val    = 1.380648813e-23;
Const(I).si.Units  = 'm^2 * kg * s^-2 * K^-1';
Const(I).cgs.Val   = 1.380648813e-16;
Const(I).cgs.Units = 'cm^2 * gr * s^-2 * K^-1';
Const(I).error     = 9.4e-9;
I = I + 1;
Const(I).Name      = 'sigma';
Const(I).Desc      = 'stefan boltzmann constant';
Const(I).Form      = '2 * pi^5 * kB^4/(15*h^3*c^2)';
Const(I).si.Val    = 5.67037321e-8;
Const(I).si.Units  = 'W * m^-2 * K^-4';
Const(I).cgs.Val   = 5.67037321e-5;
Const(I).cgs.Units = 'erg * m^-2 * K^-4';
Const(I).error     = 3.7e-8;
I = I + 1;
Const(I).Name      = 'a';
Const(I).Desc      = 'radiation constant';
Const(I).Form      = '8 * pi^5 * kB^4/(15*h^3*c^3)';
Const(I).si.Val    = 7.5657316369e-16;
Const(I).si.Units  = 'J * m^-3 * K^-4';
Const(I).cgs.Val   = 7.5657316369e-15;
Const(I).cgs.Units = 'erg * cm^-3 * K^-4';
Const(I).error     = 3.7e-8;
I = I + 1;
Const(I).Name      = 'sigmaT';
Const(I).Desc      = 'Thomson cross section';
Const(I).Form      = '8 * pi * alpha^2 * h^2 * c^2/(6*pi*me)';
Const(I).si.Val    = 6.65245854533e-29;
Const(I).si.Units  = 'm^2';
Const(I).cgs.Val   = 6.65245854533e-25;
Const(I).cgs.Units = 'cm^2';
I = I + 1;
Const(I).Name      = 're';
Const(I).Desc      = 'classical electron radius';
Const(I).Form      = 'e^2/(4*pi*eps0*me*c^2)'; % only in SI units
Const(I).si.Val    = 2.817940289458e-15;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 2.817940289458e-13;
Const(I).cgs.Units = 'cm';
Const(I).error     = 2.1e-11;
I = I + 1;
Const(I).Name      = 'a0';
Const(I).Desc      = 'bohr radius';
Const(I).Form      = 'h/(2*pi*me*c*alpha)';
Const(I).si.Val    = 5.291772109217e-11;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 5.291772109217e-9;
Const(I).cgs.Units = 'cm';
Const(I).error     = 3.2e-12;
I = I + 1;
Const(I).Name      = 'SolM';
Const(I).Desc      = 'solar mass';
Const(I).Form      = '';
Const(I).si.Val    = 1.98892e30;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 1.98892e33;
Const(I).cgs.Units = 'gr';
Const(I).error     = 1.3e-4;
I = I + 1;
Const(I).Name      = 'SolR';
Const(I).Desc      = 'solar radius';
Const(I).Form      = '';
Const(I).si.Val    = 6.96342e8;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 6.96342e10;
Const(I).cgs.Units = 'cm';
Const(I).error     = 9.3e-5;
I = I + 1;
Const(I).Name      = 'EarM';
Const(I).Desc      = 'earth mass';
Const(I).Form      = '';
Const(I).si.Val    = 5.9722e24;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 5.9722e27;
Const(I).cgs.Units = 'gr';
I = I + 1;
Const(I).Name      = 'EarR';
Const(I).Desc      = 'earth mean radius';
Const(I).Form      = '';
Const(I).si.Val    = 6371.00e3;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 6371.00e5;
Const(I).cgs.Units = 'cm';
I = I + 1;
Const(I).Name      = 'JupM';
Const(I).Desc      = 'jupiter mass';
Const(I).Form      = '';
Const(I).si.Val    = 1.8991741e27;
Const(I).si.Units  = 'kg';
Const(I).cgs.Val   = 1.8991741e30;
Const(I).cgs.Units = 'gr';
I = I + 1;
Const(I).Name      = 'JupR';
Const(I).Desc      = 'jupiter mean radius';
Const(I).Form      = '';
Const(I).si.Val    = 69911e3;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 69911e5;
Const(I).cgs.Units = 'cm';
I = I + 1;
Const(I).Name      = 'SolL';
Const(I).Desc      = 'solar luminosity';
Const(I).Form      = '';
Const(I).si.Val    = 3.839e26;
Const(I).si.Units  = 'W';
Const(I).cgs.Val   = 3.839e33;
Const(I).cgs.Units = 'erg * s^-1';
I = I + 1;
Const(I).Name      = 'pc';
Const(I).Desc      = 'parsec';
Const(I).Form      = '';
Const(I).si.Val    = 3.08567758e16;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 3.08567758e18;
Const(I).cgs.Units = 'cm';
I = I + 1;
Const(I).Name      = 'ly';
Const(I).Desc      = 'light year';
Const(I).Form      = '';
Const(I).si.Val    = 9.4607304725808e15;
Const(I).si.Units  = 'm';
Const(I).cgs.Val   = 9.4607304725808e17;
Const(I).cgs.Units = 'cm';
I = I + 1;
Const(I).Name      = 'k';
Const(I).Desc      = 'gauss gravitational constant';
Const(I).Form      = '';
Const(I).si.Val    = 0.01720209895;
Const(I).si.Units  = '';
Const(I).cgs.Val   = 0.01720209895;
Const(I).cgs.Units = '';


I = find(strcmp(lower({Const.Name}),lower(ConstantName)));
if (~isempty(I)),
   Value = Const(I).(lower(System)).Val;
   Units = Const(I).(lower(System)).Units;
else
   % look using full name
   I = find(isempty_cell(strfind(lower({Const.Desc}),lower(ConstantName)))==0);
   if (length(I)==1),
      Value = Const(I).(lower(System)).Val;
      Units = Const(I).(lower(System)).Units;
   else
      if (isempty(I)),
 	  error('Constant name was not found');
      else
 	  error('Constant name multiple possibilities');
      end
   end
end

if (nargout>3),
   for I=1:1:length(Const),
      switch lower(System)
       case 'cgs'
	   ConstS.(Const(I).Name) = Const(I).cgs.Val;
       case 'si'
	   ConstS.(Const(I).Name) = Const(I).si.Val;
       otherwise
          error('Unknown System option');
      end
   end
end
