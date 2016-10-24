function [Value,Units]=get_constant1(ConstantName,System);
%------------------------------------------------------------------------------
% get_constant1 function                                               General
% Description: Get the value of an astronomical/physical constant
%              (old version).
%              See also get_constant.m
% Input  : - Constant name:
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
%            'JupM' - Jupiter mass
%            'SolL' - Solar luminosity
%            'SolR' - Solar radius
%            'pc'   - Parsec
%            'ly'   - light year (tropical)
%            'k'    - Gaussian gravitational constant (0.017...)
%            'G-T'  - GPS-TAI time difference (19 sec)
%            'T-T'  - TT-TAI time difference (32.184 sec)
%            'JY'   - Julian year (365.25 days).
%          - Unit system:
%            'SI'  - meter-kg-sec
%            'cgs' - cm-gram-sec, default
%            'agd' - au-gram-day
% Output : - Constant value.
%          - Constant units.
% Tested : Matlab 6.5
%     By : Eran O. Ofek                      July 2003
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [c,Units]=get_constant('c','cgs');  % get the speed of light
% ToDo: http://physics.nist.gov/cuu/Constants/Table/allascii.txt
% Reliable: 1
%------------------------------------------------------------------------------
if (nargin==1),
   System = 'cgs';
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end


switch ConstantName
 case 'au'
    %--- DE405 (Standish 1998) ---
    switch System
     case 'SI'
        Value = 1.49597870691e11;    % [m]
        Units = 'm';
     case 'cgs'
        Value = 1.49597870691e13;    % [cm]
        Units = 'cm';
     case 'agd'
        Value = 1;                   % [au]        
        Units = 'au';
     otherwise
        error('Unknwon unit system');
    end
       
 case 'G'
    switch System
     case 'SI'
        Value = 6.67259e-11;   % [m^3 kg^-1 s^-2]
        Units = 'm^3 * kg^-1 * s^-2';
     case 'cgs'
        Value = 6.67259e-8;    % [cm^3 gr^-1 s^-2]
        Units = 'cm^3 * gr^-1 * s^-2';
     case 'agd'
        Value = 1.4878039e-37; % [au^3 gr^-1 day^-2]   
        Units = 'au^3 * gr^-1 * day^-2';
     otherwise
        error('Unknwon unit system');
    end
   
 case 'c'
    switch System
     case 'SI'
        Value = 299792458;        % [m s^-1]
        Units = 'm * s^-1';
     case 'cgs'
        Value = 29979245800;      % [cm s^-1]
        Units = 'cm * s^-1';
     case 'agd'
        Value = 173.144632720536; % [au day^-1]      
        Units = 'au * day^-1';
     otherwise
        error('Unknwon unit system');
    end

 case 'h'
    switch System
     case 'SI'
        Value = 6.6260755e-34;         % [kg m^2 s^-1]
        Units = 'm^2 * kg * s^-1';
     case 'cgs'
        Value = 6.6260755e-27;         % [gr cm^2 s^-1]
        Units = 'cm^2 * gr * s^-1';
     case 'agd'
        Value = 2.55811048951848e-48;  % [gr au^2 day^-1]      
        Units = 'au^2 * gr * day^-1';
     otherwise
        error('Unknwon unit system');
    end

 case 'e'
    switch System
     case 'SI'
        Value = 1.60217733e-19;    % [C]
        Units = 'C';
     case 'cgs'
        Value = 4.8032068e-10;     % [esu]
        Units = 'esu';
     case 'agd'
        error('Avialable only in SI or cgs units');
     otherwise
        error('Unknwon unit system');
    end

 case 'alpha'
    switch System
     case 'SI'
        Value = 7.29735308e-3;   % []
        Units = '';
     case 'cgs'
        Value = 7.29735308e-3;   % []
        Units = '';
     case 'agd'
        Value = 7.29735308e-3;   % []
        Units = '';
     otherwise
        error('Unknwon unit system');
    end

 case 'Ryd'
    switch System
     case 'SI'
        Value = 10973731.534;       % [m^-1]
        Units = 'm^-1';
     case 'cgs'
        Value = 109737.31534;       % [cm^-1]
        Units = 'cm^-1';
     case 'agd'
        Value = 1.64164687102e+18;  % [au^-1]
        Units = 'au^-1';
     otherwise
        error('Unknwon unit system');
    end

 case 'mp'
    switch System
     case 'SI'
        Value = 1.6726231e-27;      % [kg]
        Units = 'kg';
     case 'cgs'
        Value = 1.6726231e-24;      % [gr]
        Units = 'gr';
     case 'agd'
        Value = 1.6726231e-24;      % [gr]
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end

 case 'me'
    switch System
     case 'SI'
        Value = 9.10938997e-31;      % [kg]
        Units = 'kg';
     case 'cgs'
        Value = 9.10938997e-28;      % [gr]
        Units = 'gr';
     case 'agd'
        Value = 9.10938997e-28;      % [gr]
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end

 case 'amu'
    switch System
     case 'SI'
        Value = 1.6605402e-27;       % [kg]
        Units = 'kg';
     case 'cgs'
        Value = 1.6605402e-24;       % [gr]
        Units = 'gr';
     case 'agd'
        Value = 1.6605402e-24;       % [gr]
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end

 case 'NA'
    switch System
     case 'SI'
        Value = 6.0221367e23;           % [mol^-1]
        Units = 'mol^-1';
     case 'cgs'
        error('Avialable only in SI units');
     case 'agd'
        error('Avialable only in SI units');
     otherwise
        error('Unknwon unit system');
    end    

 case 'kB'
    switch System
     case 'SI'
        Value = 1.380658e-23;         % [J K^-1 = kg m^2 s^-2 K^-1]
        Units = 'm^2 * kg * s^-2 * K^-1';
     case 'cgs'
        Value = 1.380658e-16;         % [dyn K^-1 = gr cm^2 s^-2 K^-1]
        Units = 'cm^2 * gr * s^-2 * K^-1';
     case 'agd'
        Value = 4.605351411e-33;      % [gr au^2 day^-2 K^-1]
        Units = 'au^2 * gr * dat^-2 * K^-1';
     otherwise
        error('Unknwon unit system');
    end

 case 'sigma'
    switch System
     case 'SI'
        Value = 5.67051e-8;           % [W m^-2 K^-4]
        Units = 'W * m^-2 * K^-4';
     case 'cgs'
        Value = 5.67051e-5;           % [erg cm^-2 K^-4 s^-1]
        Units = 'erg * m^-2 * K^-4';
     case 'agd'
        Value = NaN;
     otherwise
        error('Unknwon unit system');
    end

 case 'sigmaT'
    switch System
     case 'SI'
        Value = 6.6524e-29;           % [m^2]
        Units = 'm^2';
     case 'cgs'
        Value = 6.6524e-25;           % [cm^2]
        Units = 'cm^2';
     case 'agd'
        Value = NaN;                  % [au^2]
        Units = 'au^2';
     otherwise
        error('Unknwon unit system');
    end

 case 'BohrR'
    % Bohr radius.
    Value = 0.52917720859e-10;    % [m]
    switch System
     case 'SI'
        Value = Value;                % [m]
        Units = 'm';
     case 'cgs'
        Value = Value.*100            % [cm]
        Units = 'cm';
     case 'agd'
        AU = get_constant('au','SI');
        Value = Value./AU             % [au]
        Units = 'au';   
     otherwise
        error('Unknwon unit system');
    end


 case 'SolM'
    switch System
     case 'SI'
        Value = 1.988435e30;      % [kg] (+/-27)
        Units = 'kg';
     case 'cgs'
        Value = 1.988435e33;        % [gr]
        Units = 'gr';
     case 'agd'
        Value = 1.988435e33;        % [gr]      
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end
    
 case 'EarM'
    switch System
     case 'SI'
        Value = 5.97424e24;       % [kg]
        Units = 'kg';
     case 'cgs'
        Value = 5.97424e27;       % [gr]
        Units = 'gr';
     case 'agd'
        Value = 5.97424e27;       % [gr]      
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end
    
 case 'JupM'
    switch System
     case 'SI'
        Value = 1.8991741e27;       % [kg]
        Units = 'kg';
     case 'cgs'
        Value = 1.8991741e30;       % [gr]
        Units = 'gr';
     case 'agd'
        Value = 1.8991741e30;       % [gr]      
        Units = 'gr';
     otherwise
        error('Unknwon unit system');
    end

 case 'SolL'
    switch System
     case 'SI'
        Value = 3.827e26;         % [watt]
        Units = 'W';
     case 'cgs'
        Value = 3.827e33;         % [erg]
        Units = 'erg';
     case 'agd'
        Value = NaN;              %
     otherwise
        error('Unknwon unit system');
    end

 case 'SolR'
    switch System
     case 'SI'
        Value = 6.95508e8;         % [m]
        Units = 'm';
     case 'cgs'
        Value = 6.95508e10;            % [cm]
        Units = 'cm';
     case 'agd'
        Value = 0.00464918382051438;    % [au] 
        Units = 'au';
     otherwise
        error('Unknwon unit system');
    end

    
 case 'pc'
    switch System
     case 'SI'
        Value = 3.08567758e16;       % [m]
        Units = 'm';
     case 'cgs'
        Value = 3.08567758e18;       % [cm]
        Units = 'cm';
     case 'agd'
        Value = 206264.806247;       % [au]      
        Units = 'au';
     otherwise
        error('Unknwon unit system');
    end

 case 'ly'
    switch System
     case 'SI'
        Value = 9.4607304725808e15;       % [m]
        Units = 'm';
     case 'cgs'
        Value = 9.4607304725808e17;       % [cm]
        Units = 'cm';
     case 'agd'
        Value = 63241.077101176;          % [au]      
        Units = 'au';
     otherwise
        error('Unknwon unit system');
    end

 case 'k'
    Value   = 0.01720209895;

 case 'G-T'
    switch System
     case 'SI'
        Value = 19;                    % [sec]
        Units = 's';
     case 'cgs'
        Value = 19;                    % [sec]
        Units = 's';
     case 'agd'
        Value = 0.000219907407407407;  % [day]   
        Units = 'day';
     otherwise
        error('Unknwon unit system');
    end

 case 'T-T'
    switch System
     case 'SI'
        Value = 32.184;                    % [sec]
        Units = 's';
     case 'cgs'
        Value = 32.184;                    % [sec]
        Units = 's';
     case 'agd'
        Value = 0.0003725;                 % [day]   
        Units = 'day';
     otherwise
        error('Unknwon unit system');
    end

 case 'JY'
    switch System
     case 'SI'
        Value = 365.25.*86400;                    % [sec]
        Units = 's';
     case 'cgs'
        Value = 365.25.*86400;                    % [sec]
        Units = 's';
     case 'agd'
        Value = 365.25;                           % [day]   
        Units = 'day';
     otherwise
        error('Unknwon unit system');
    end

    
 otherwise
    error('Unknwon constant name');
end




