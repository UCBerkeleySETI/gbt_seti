function [X,Y]=pr_stereographic_polar(Az,ZenithDist);
%------------------------------------------------------------------------------
% pr_stereographic_polar function                                     AstroMap
% Description: Project coordinates (longitude and latitude) using the
%              Stereographic polar projection.
%              This projection preservs angles.
% Input  : - Vector of Azimuth, in radians.
%          - Vector of Zenith-distance, in radians.
% Output : - Vector of X position
%          - Vector of Y position 
% Tested : Matlab 5.3
%     By : Eran O. Ofek                  November 2004  
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------
if (nargin==2),
   % no default
elseif (nargin==3),
   error('Illigal number of argument');
end

X     = cos(Az).*tan(0.5.*ZenithDist);
Y     = sin(Az).*tan(0.5.*ZenithDist);


