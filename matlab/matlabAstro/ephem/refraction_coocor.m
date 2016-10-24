function [DelAlpha,DelDelta]=refraction_coocor(RA,Dec,Ref,varargin);
%------------------------------------------------------------------------------
% refraction_coocor function                                             ephem
% Description: Calculate the correction in equatorial coordinates due to
%              atmospheric refraction.
% Input  : - Vector of R.A. in radians, [H M S] or sexagesimal format.
%            See convertdms.m for details.
%          - Vector of Dec. in radians, [Sign D M S] or sexagesimal format.
%          - Refraction in radians. See refraction.m.
%          - Parallactic angle in radians. See parallactic_angle.m.
%            Alternatively, if additional input argument are specified
%            then this is assumed to be the JD.
%          - Optional Geodetic position [Long, Lat], in radians.
%            If specified then the previous input argument is assumed
%            to be JD.
% Output : - Offset in R.A. needed to be added to true position in
%            order to get the apparent position [radians].
%          - Offset in Dec. needed to be added to true position in
%            order to get the apparent position [radians].
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    October 2010
%     URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Examples: [DelAlpha,DelDelta]=refraction_coocor(RA,Dec,Ref,ParAng);
%           [DelAlpha,DelDelta]=refraction_coocor(RA,Dec,Ref,JD,GeodPos);
% Reliable: 1
%------------------------------------------------------------------------------

RA_rad  = convertdms(RA,'gH','r');
Dec_rad = convertdms(Dec,'gD','R');

Narg = length(varargin);
switch Narg
 case 1
    % assume input is parallactic angle
    ParAng = varargin{1};
 case 2
    % assume input is JD, GeodPos
    JD      = varargin{1};
    GeodPos = varargin{2}; 

    LST     = lst(JD,GeodPos(1));  % mean local sidereal time
    ParAng  = parallactic_angle([RA_rad, Dec_rad],LST,GeodPos(2));
 otherwise
    error('Illegal number of input arguments');
end

DelAlpha = Ref.*sec(Dec_rad).*sin(ParAng);
DelDelta = Ref.*cos(ParAng);



