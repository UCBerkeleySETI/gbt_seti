function Rho=matter_density(OmegaM,H0)
%--------------------------------------------------------------------------
% matter_density function                                        cosmology
% Description: Calculate the mean matter density in the Universe.
% Input  : - OmegaM, or cosmological parameteters. See cosmo_pars.m
%            for details. Default is 'wmap9'.
%          - If the first parameter is OmegaM than this should be H0.
% Output : - Density [gr/cm^3]
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Jul 2006
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Rho=matter_density(0.3,70); Rho=matter_density;
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==0),
    OmegaM = 'wmap9';
end
if (ischar(OmegaM)),
    Par = cosmo_pars('wmap9');
else
    Par.OmegaM = OmegaM;
    Par.H0     = H0;
end

G  = get_constant('G','cgs');
Pc = get_constant('pc','cgs');
% conver H0 to cgs
Par.H0 = Par.H0.*1e5./(Pc.*1e6);

Rho = Par.OmegaM.*3.*Par.H0.^2./(8.*pi.*G);
