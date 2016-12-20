function [Run,MJD,Dist]=coo2run(RA,Dec,DR,Assign)
%------------------------------------------------------------------------------
% coo2run function                                                        sdss
% Description: Given celestial equatorial coordinates, find all SDSS
%              Run/Rerun/Col/Field number ID that cover the coordinates.
% Input  : - J2000.0 RA in [rad] or [H M S] or sexagesimal string.
%          - J2000.0 Dec in [rad] or [Sign D M S] or sexagesimal string.
%          - SDSS data relaese, default is 'DR9'.
%            If empty (i.e., []), then use default.
%          - Assign and read {0 | 1} the list of fields into/from
%            the matlab workspace.
%            1 - yes; 0 - no. Default is 0.
%            If you are running coo2run many times, use 1 for faster
%            execuation.
% Output : - A cell array of lists of [Run, Rerun, Col, Field]
%            that covers the given coordinates. Ecah cell for each
%            coordinate.
%          - A cell array of modified JD of frame for [u g r i z] frames,
%            one line per run. Each cell per each coordinate.
%          - A cell array od distances (in radians) of the requested
%            RA, Dec from each side of the frame polygon.
%            Each cell per each coordinate.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                     March 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Package: sdss
% Example: [Run,MJD,D]=coo2run([10 04 00],[1 41 0 0]);
%          min(D{1}.*RAD.*3600,[],2)  % print the distance ["] of the
%                                     % coordinate from nearest boundry.
% Bugs   : doesn't handle RA 0-2pi jump
% Reliable: 2
%------------------------------------------------------------------------------

RAD    = 180./pi;
ColRA  = 10;
ColDec = 14;
Threshold  = 0.3./RAD;   % search threshold (larger than frame size)

DefDR = 'DR9';
if (nargin==2),
   DR     = DefDR;
   Assign = 0;
elseif (nargin==3),
   Assign = 0;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(DR)==1),
   DR = DefDR;
end

RA  = convertdms(RA,'gH','r');
Dec = convertdms(Dec,'gD','R');

switch Assign
 case 1
    try
       Fields = evalin('base','Fields;');
    catch

       %--- Load SDSS Fields ---
       FieldsStr = sprintf('SDSS_%s_Fields_All_PolySort',DR);

       eval(sprintf('load %s.mat',FieldsStr));
       eval(sprintf('Fields = %s;',FieldsStr));   % copy to Fields
       eval(sprintf('clear %s;',FieldsStr));

       assignin('base','Fields',Fields);
    end
 otherwise
    %--- Load SDSS Fields ---
    FieldsStr = sprintf('SDSS_%s_Fields_All_PolySort',DR);

    eval(sprintf('load %s.mat',FieldsStr));
    eval(sprintf('Fields = %s;',FieldsStr));   % copy to Fields
    eval(sprintf('clear %s;',FieldsStr));
end


Ind  = [1 2; 2 3; 3 4; 4 1];   % indecies of sides
N    = length(RA);
Run  = cell(N,1);
MJD  = cell(N,1);
Dist = cell(N,1);
for I=1:1:N,
   L = cat_search(Fields,[ColRA ColDec],[RA(I) Dec(I)],Threshold,'circle','Dec','sphere');

   InL = [];
   for J=1:1:length(L),
      Inside1 = inpolygon(RA(I),Dec(I),Fields(L(J),ColRA+[0:1:3]), Fields(L(J),ColDec+[0:1:3]));
      Inside2 = inpolygon(RA(I)-2.*pi,Dec(I),Fields(L(J),ColRA+[0:1:3]), Fields(L(J),ColDec+[0:1:3]));
      Inside3 = inpolygon(RA(I)+2.*pi,Dec(I),Fields(L(J),ColRA+[0:1:3]), Fields(L(J),ColDec+[0:1:3]));
      Inside  = sign(Inside1 + Inside2 + Inside3);
      if (Inside==1),
         InL = [InL; L(J)];
      end
   end
   Run{I} = Fields(InL,1:4);
   MJD{I} = Fields(InL,5:9);

   %--- calculate distance of point from boundries ---
   % assuming plane(!) geometry (approximation)
   for Is=1:1:length(Ind),
      ColRA1  = ColRA  - 1 + Ind(Is,1);
      ColRA2  = ColRA  - 1 + Ind(Is,2);
      ColDec1 = ColDec - 1 + Ind(Is,1);
      ColDec2 = ColDec - 1 + Ind(Is,2);

      D0 = sphere_dist(Fields(InL,ColRA1), Fields(InL,ColDec1), Fields(InL,ColRA2), Fields(InL,ColDec2));
      D1 = sphere_dist(Fields(InL,ColRA1), Fields(InL,ColDec1), RA(I), Dec(I));
      D2 = sphere_dist(Fields(InL,ColRA2), Fields(InL,ColDec2), RA(I), Dec(I));

      S = 0.5.*(D0+D1+D2);    % semi-perimeter
      H = 2.*sqrt(S.*(S-D0).*(S-D1).*(S-D2))./D0;  % Height using Heron formula
      Dist{I}(:,Is) = H;
   end
end
clear Fields;
