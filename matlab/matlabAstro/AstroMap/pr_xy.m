function [X,Y]=pr_xy(Long,Lat,R);
%------------------------------------------------------------------------------
% pr_xy function                                                      AstroMap
% Description: Project coordinates (longitude and latitude) to X-Y
%              projection (no transformation).
% Input  : - Vector of Longitude, in radians.
%          - Vector of Latitude, in radians.
%          - Scale radius, default is 1.
% Output : - Vector of X position
%          - Vector of Y position 
% Tested : Matlab 5.2
%     By : Eran O. Ofek                    August 1999     
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------

if (nargin==3),
   % no default
elseif (nargin==2),
   % no default
   R = 1;
else
   error('Illigal number of argument');
end


X = R.*Long./(2.*pi);
Y = R.*Lat./(pi);



