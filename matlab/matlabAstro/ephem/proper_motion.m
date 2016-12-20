function varargout=proper_motion(EpochOut,EpochIn,RA,Dec,PM_RA,PM_Dec,Parallax,RadVel);
%------------------------------------------------------------------------------
% proper_motion function                                                 ephem
% Description: Applay proper motion to a acatalog
% Input  : - Final epoch (days).
%          - Catalog initial epoch (days).
%          - RA at initial epoch [radians].
%          - Dec at initial epoch [radians].
%          - Proper motion in RA [mas/yr].
%          - Proper motion in Dec [mas/yr].
%          - Parallax [mas], default is 1e-3;
%          - Radial velocity [km/s], default is 0.
% Output * Either 2 or 3 output arguments.
%          If two output arguments, these are the final RA and Dec
%          in radians.
%          [RA,Dec] = proper_motion(...);
%          If three output arguments, these are the X,Y,Z cosine directions
%          in units of AU.
%          [X,Y,Z] = proper_motion(...);
% Tested : Matlab 7.11
%     By : Eran O. Ofek                   January 2011
%    URL : http://wise-obs.tau.ac.il~/eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------

Def.Parallax = 1e-3;
Def.RadVel   = 0;
if (nargin==6),
   Parallax = Def.Parallax;
   RadVel   = Def.RadVel;
elseif (nargin==7),
   RadVel   = Def.RadVel;
elseif (nargin==8),
   % do nothing
else
   error('Illegal number of input arguments');
end


[Rdot,R] = pm2space_motion(RA,Dec,PM_RA,PM_Dec,Parallax,RadVel);
Rn       = R + Rdot.*(EpochOut - EpochIn);
if (nargout==2),
   [varargout{1},varargout{2}] = cosined2coo(Rn(:,1),Rn(:,2),Rn(:,3));
elseif (nargout==3),
   varargout{1} = Rn(:,1);
   varargout{2} = Rn(:,2);
   varargout{3} = Rn(:,3);
end
