function [Prof,VecR,VecX,VecY]=lineprof(Image,Start,End,Width,Step,Stat,Interp)
%--------------------------------------------------------------------------
% lineprof function                                                ImBasic
% Description: Given two coordinates in a matrix the script return the
%              intensity as function of position along the line between
%              the two points.
% Input  : - Matrix.
%          - Start point [x, y].
%          - End point [x, y].
%          - Width of the line (odd integer), default is 1.
%          - Approximate step size along the line [pixels], default is 1.
%            The actual step size is adjusted so nultiply it by an integer
%            will give the length of the line.
%          - Value to calculate:
%            'sum' | 'mean' | 'median' | 'std' | 'min' | 'max',
%            default is 'mean'.
%          - Interpolation method (see interp1), default is 'linear'
% Output : - Vector of profile.
%          - Vector of position along the line [pixels]
%          - Corresponding X position [pixels]
%          - Corresponding Y position [pixels]
% Tested : Matlab 5.3
%     By : Eran O. Ofek                    Feb 2004
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Image = rand(1000,1000);
%          [Prof,VecR,VecX,VecY]=lineprof(Image,[100 200],[200 300]);
% Reliable: 2
%--------------------------------------------------------------------------
DefWidth    = 1;
DefStep     = 1;
DefStat     = 'mean';
DefInterp   = 'linear';

if (nargin==3),
   Width    = DefWidth;
   Step     = DefStep;
   Stat     = DefStat;
   Interp = DefInterp;
elseif (nargin==4),
   Step     = DefStep;
   Stat     = DefStat;
   Interp = DefInterp;
elseif (nargin==5),
   Stat     = DefStat;
   Interp = DefInterp;
elseif (nargin==6),
   Interp = DefInterp;
elseif (nargin==7),
   % no default
else
   error('Illegal number of input arguments');
end

if (Width~=floor(Width) | floor(0.5.*(Width+1))~=0.5.*(Width+1)),
   error('Width should be an odd integer');
end

Size   = size(Image);
ImX    = [1:1:Size(2)].';
ImY    = [1:1:Size(1)].';
% Read Image to memory
X(1)   = Start(1);
X(2)   = End(1);
Y(1)   = Start(2);
Y(2)   = End(2);

DiffX  = diff(X);
DiffY  = diff(Y); 
SlopeL = atan2(DiffY,DiffX);   % slope of the line
SlopeV = atan2(DiffX,DiffY);   % slope of vertical to line

Length = sqrt(DiffX.^2+DiffY.^2);
Nstep  = ceil(Length./Step);
Step   = Length./Nstep;
VecR   = [0:Step:Length].';
VecX   = [X(1):DiffX./Nstep:X(2)].';
if (isempty(VecX)),
   VecX = X(1);
end
VecY   = [Y(1):DiffY./Nstep:Y(2)].';
if (isempty(VecY)),
   VecY = Y(1);
end
HalfWidth = 0.5.*(Width - 1);

Data = zeros(Width,Nstep+1);
K    = 0;
for PosWidth=-HalfWidth:1:HalfWidth,
   K       = K + 1;
   CurX    = X + PosWidth.*cos(SlopeV);
   CurY    = Y - PosWidth.*sin(SlopeV);
   
   CurVecX = [CurX(1):DiffX./Nstep:CurX(2)].';
   CurVecY = [CurY(1):DiffY./Nstep:CurY(2)].';
   if (isempty(CurVecX)),
      CurVecX = CurX(1);
   end
   if (isempty(CurVecY)),
      CurVecY = CurY(1);
   end

   Data(K,:) = interp2(ImX,ImY,Image,CurVecX,CurVecY,Interp).';

end


switch Stat
 case 'mean'
    Prof = mean(Data,1);
 case 'std'
    Prof = std(Data,0,1);
 case 'median'
    Prof = median(Data,1);
 case 'sum'
    Prof = sum(Data,1);
 case 'min'
    Prof = min(Data,[],1);
 case 'max'
    Prof = max(Data,[],1);
 otherwise
    error('Unknown statistics option');
end
