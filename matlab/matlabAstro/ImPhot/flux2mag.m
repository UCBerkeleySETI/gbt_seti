function Mag=flux2mag(Flux,ZP,Luptitude,Soft)
%--------------------------------------------------------------------------
% flux2mag function                                                 ImPhot
% Description: Convert flux to magnitude or luptitude. 
% Input  : - Flux
%          - ZP
%          - A flag indicating if to return Luptitude (true) or
%            magnitude (false). Default is true.
%          - Luptitude softening parameter (B), default is 1e-10.
% Output : - Magnitude or luptitude.
% License: GNU general public license version 3
% See also: luptitude.m
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Mag=flux2mag(1,1)
% Reliable: 2
%--------------------------------------------------------------------------

Def.ZP        = 0;
Def.Luptitude = true;
Def.Soft      = 1e-10;
if (nargin==1),
    ZP        = Def.ZP;
    Luptitude = Def.Luptitude;
    Soft      = Def.Soft;
elseif (nargin==2),
    Luptitude = Def.Luptitude;
    Soft      = Def.Soft;
elseif (nargin==3),
    Soft      = Def.Soft;
elseif (nargin==4),
    % do nothing
else
    error('Illegal number of input arguments');
end


if (Luptitude),
    Mag = luptitude(Flux,10.^(0.4.*ZP),Soft);
else
    Mag = ZP - 2.5.*log10(Flux);
end
