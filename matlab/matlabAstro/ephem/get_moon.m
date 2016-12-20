function Moon=get_moon(JD,GeodCoo)
%--------------------------------------------------------------------------
% get_moon function                                                  ephem
% Description: Get Moon position (low accuracy).
% Input  : - JD, or date (see jd.m for available formats).
%          - Geodetic coordinates [Long, Lat] in radians.
% Output : - Structure containing Moon position, with the following fields:
%            .RA   - RA [radians]
%            .Dec  - Dec [radians]
%            .Az   - Azimuth [radians]
%            .Alt  - Altitude [radians]
%            .IllF - Illuminated fraction
%            .Phase- Moon phase angle [radians]
% Tested : Matlab 7.3
%     By : Eran O. Ofek                    Jun 2008
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 1
%--------------------------------------------------------------------------

if (size(JD,2)==1),
   % already in JD
else
  JD = julday(JD).';
end

[Moon.RA, Moon.Dec] = mooncool(JD,GeodCoo,'b');
I = find(Moon.RA<0);
if (isempty(I)==1),
   % do nothing
else
   Moon.RA(I) = 2.*pi + Moon.RA(I);
end
HorizCoo = horiz_coo([Moon.RA, Moon.Dec],JD,GeodCoo,'h');
Moon.Az   = HorizCoo(:,1);
Moon.Alt  = HorizCoo(:,2);


[Moon.IllF,Moon.Phase] = moon_illum(JD);
