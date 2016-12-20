function DA=ad_q_dist(Z,CosmoPars);
%------------------------------------------------------------------------------
% ad_q_dist function                                                 csomology
% Description: Compute filled-beam angular-diameter distance to an object
%              in a flat Universe with constant equation of state
%              (p=w\rho; i.e., quintessence).
% Input  : - Vector of redshifts, if two column matrix is given,
%            then integrate the distance from the value in the first column
%            to the value in the second column.
%          - Cosmological parameters : [H0, \Omega_{m}, w],
%            or cosmological parmeters structure, or a string containing
%            parameters source name, default is 'wmap3' (see cosmo_pars.m).
% Output : - Angular-diameter distance [parsec].
% Reference : Perlmutter et al. 1997 ApJ, 483, 565
%             Jain et al. (2002), astro-ph/0105551
% Notes  : Previously called: ad_fq_dist.m
% Tested : Matlab 5.3
%     By : Eran O. Ofek                      June 2002
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: ad_q_dis(1);
% Reliable: 2
%------------------------------------------------------------------------------
C  = 29979245800;    % speed of light [cm/sec]
Pc = 3.0857e18;      % Parsec [cm]

if (nargin==1),
   CosmoPars = 'wmap3';
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end


if (isstr(CosmoPars)==0 & isstruct(CosmoPars)==0),
   % do nothing
else
   Par = cosmo_pars(CosmoPars);
   CosmoPars = [Par.H0, Par.OmegaM, Par.W];
end


if (size(Z,2)==1),
   Z = [zeros(size(Z)), Z];
end


H0     = CosmoPars(1);
OmegaM = CosmoPars(2);
W      = CosmoPars(3);


% convert H0 to cm/sec/sec
H0 = H0*100000/(Pc*1e6);
R0 = C./H0;


%--- Note : Assumes a flat universe ---

N  = size(Z,1);
DA = zeros(size(Z,1),1);
for I=1:1:N,
   ZI1 = Z(I,1);
   ZI2 = Z(I,2);

   Int = inline('1./sqrt(OmegaM.*(1+ZI).^3 + (1-OmegaM).*(1+ZI).^(3.*(1+W)))','ZI','OmegaM','W');

   DA_Int = quad(Int,ZI1,ZI2,[],[],OmegaM,W);

   DA(I) = R0.*DA_Int./(1+ZI2);

end

% luminosity distance in Parsecs:
DA = DA./Pc;


