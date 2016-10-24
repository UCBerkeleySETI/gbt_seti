function [Tot_Illum, Sun_Illum, Moon_Illum]=skylight(JD,GeodPos);
%------------------------------------------------------------------------------
% function skylight                                                      ephem
% Description: Calculate the total sky illumination due to the Sun, Moon,
%              stars and air-glow, in Lux on horizontal surface as a 
%              function of time.
% Input  : - vector of JD.
%          - Geodetic position [East_Long, North_Lat] in radians.
% Output : - Total illumination in Lux on horiz. surface.
%          - Sun+sky illumination in Lux on horiz. surface.
%          - Moon illumination in Lux on horiz. surface.
% Tested : Matlab 5.3
%     By : Eran O. Ofek                    August 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------

[Sun_RA, Sun_Dec]            = suncoo(JD);
[Moon_RA, Moon_Dec, Moon_HP] = mooncool(JD,GeodPos);

Moon_Elon = acos(sin(Sun_Dec).*sin(Moon_Dec) + cos(Sun_Dec).*cos(Moon_Dec).*cos(Sun_RA-Moon_RA));

Sun_Horiz  = horiz_coo([Sun_RA,Sun_Dec],JD,GeodPos,'h');
Moon_Horiz = horiz_coo([Moon_RA,Moon_Dec],JD,GeodPos,'h');
% calculate sunlight + starlight + airglow
Sun_Illum  = sunlight(Sun_Horiz(:,2));

% calculate moonlight
Moon_Illum = moonlight(Moon_Horiz(:,2),Moon_HP,Moon_Elon);

Tot_Illum = Sun_Illum + Moon_Illum;

