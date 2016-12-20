function jupiter_map(Par)
%--------------------------------------------------------------------------
% jupiter_map function                                               ephem
% Description: Plot Jupiter image as observed from Earth at a given time.
% Input  : - If two elements vector then:
%            [long_of_Sys_I, long_of_Sys_II]
%            else JD (scalar), or date vector (single date;
%            see julday.m for options). Date in TT time scale.
% Output : null
% Plot   : Jupiter RGB image refer to Jupiter System II, illuminated disk.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Jan 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Needed : Map of Jupiter JupiterVoyagerMap.jpg (source: Voyager)
% Example: jupiter_map(2451545);
% Web example: http://astroclub.tau.ac.il/ephem/JovMap/
% Reliable: 2
%--------------------------------------------------------------------------
JupiterMapFile = 'JupiterVoyagerMap.jpg';

if (length(Par)==2),
   CMi = Par;
else
   if (length(Par)>1),
      JD = julday(Par);
   else
      JD = Par;
   end
   [CMg,CMi,Ds,De]=jup_meridian(JD);
end

FloorDeg = inline('360.*(X./360 - floor(X./360))','X');

%
% Assuming the red spot is at long. 109
%
SysI  = FloorDeg(CMi(1)+180); % 195
SysII = FloorDeg(CMi(2)+180); % 195


Im = imread(JupiterMapFile);
[SizeIm] = size(Im);
Scale = SizeIm(1)./180;

axesm ortho;
geoshow(Im, [Scale 90 SysII])
axis off;
set(gca,'YDir','Reverse');

