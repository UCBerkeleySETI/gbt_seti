function [VecX,VecY,VecR,VecVal,Par]=ds9_getvecprof(CooX,CooY,varargin)
%--------------------------------------------------------------------------
% ds9_getvecprof function                                              ds9
% Description: Given the X and Y coordinates of two points, and an open 
%              ds9 display, get the value of the image in the display,
%              interpolated along the line connecting the two points.
% Input  : - X coordinates of the fisrt and second point.
%          - Y coordinates of the fisrt and second point.
%          * Arbitrary number of pairs of ...,key,val,... arguments.
%            The following keywords are available:
%            'CooType'  - Coordinates type in ds9 {'image'}. 
%                         Default is 'image'.
%            'VecStep'  - Interpolation step for "v" option.
%                         Default is 0.3.
%            'InterpMethod' - Interpolation method. Default is 'linear'.
% Output : - 
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [VecX,VecY,VecR,VecVal,Par]=ds9_getvecprof(CooX,CooY);
% Reliable: 2
%--------------------------------------------------------------------------


DefV.VecStep      = 0.3;
DefV.InterpMethod = 'linear';
DefV.CooType      = 'image';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


Nstep = max(range(CooX),range(CooY))./InPar.VecStep;
BoxCoo = [floor(min(CooX)), floor(min(CooY)),...
          ceil(max(CooX)) - floor(min(CooX))+1,...
          ceil(max(CooY)) - floor(min(CooY))+1];
[MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
                    
Par = polyfit(CooX,CooY,1);
                    
VecX = linspace(CooX(1),CooX(2),Nstep).';
VecY = polyval(Par,VecX);
%SignY          = sign(Exam(Ind).CooY(2) - Exam(Ind).CooY(1));
%Exam(Ind).VecY = (Exam(Ind).CooY(1):SignY.*InPar.VecStep:Exam(Ind).CooY(2)).';
VecVal  = interp2(MatX,MatY,MatVal,VecX,VecY,InPar.InterpMethod);
VecR = sqrt( (VecX - min(VecX)).^2 + ...
             (VecY - min(VecY)).^2);
                    