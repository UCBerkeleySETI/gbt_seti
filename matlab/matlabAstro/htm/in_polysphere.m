function Flag=in_polysphere(Positions,Corners,Crit)
%--------------------------------------------------------------------------
% in_polysphere function                                               htm
% Description: Check if a list of positions are found within a convex
%              polygon on the celestial sphere in which its sides are
%              great circles.
%              The polygon should be defined according to the
%              right-hand rule.
% Input  : - List of positions to check if they are inside the convex
%            spherical polygon. Each row corrsponds to a position.
%            This is a matrix of either 2 or 3 columns.
%            If two columns are provided then these are [Long, Lat]
%            in radians. If three columns are given, these are
%            cosine directions.
%          - The verteces of a convex polygon on the celestial sphere
%            in whichits sides are great circles.
%            Each row correspond to one vertex.
%            This is a matrix of either 2 or 3 columns.
%            If two columns areprovided then these are [Long, Lat]
%            in radians. If three columns are given, these are
%            cosine directions.
%            The coordinates should be ordered such that the
%            right-hand rule is pointing toward the
%            center of the polygon.
%          - Flag indicating if to use ">" or ">=" in in_halfspace.m.
%            If 1 (default) then use (N dot R > C),
%            If 2 then use (N dot R) >= C
% Output : - A flag indicating if each position (row in Positions
%            matrix) is inside (true) or on/outside the polygon (false).
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Jul 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Corners=[0 0;1 0;1 1;0 1]
%          Positions=[0.5 0.5;2 2; 0 0; eps eps];
%          Flag = in_polysphere(Positions,Corners);
% Reliable: 2
%--------------------------------------------------------------------------

Def.Crit = 1;
if (nargin==2),
   Crit  = Def.Crit;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (size(Positions,2)==2),
   [CD1, CD2, CD3] = coo2cosined(Positions(:,1),Positions(:,2));
   Positions       = [CD1, CD2, CD3];
end

if (size(Corners,2)==2),
   [CD1, CD2, CD3] = coo2cosined(Corners(:,1),Corners(:,2));
   Corners         = [CD1, CD2, CD3];
end

Corners = [Corners;Corners(1,:)];
% for each pair of verteces call in_halfspace.m
Nvert = size(Corners,1)-1;
% cross (X) two verteces in order to get the pole of the great
% circle defined by the two verteces
PoleVec = cross_fast(Corners(1:end-1,:),Corners(2:end,:));
% normalize polar vector to unity
PoleVec = bsxfun(@times,PoleVec, 1./sqrt(sum(PoleVec.^2,2)));
FlagMat = in_halfspace(Positions,PoleVec,0,1,Crit).';
Flag    = sum(FlagMat,2)==Nvert;
