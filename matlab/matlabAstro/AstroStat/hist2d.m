function [Mat,VecX,VecY]=hist2d(Xv,Yv,RangeX,StepX,RangeY,StepY)
%--------------------------------------------------------------------------
% hist2d function                                                AstroStat
% Description: calculate the 2-D histogram of 2-D data set.
% Input  : - Vector of X coordinates.
%          - Vector of Y coordinates.
%          - Range of X histogram [min max].
%          - Step size of X histogram.
%          - Range of Y histogram [min max].
%          - Step size of Y histogram.
% Output : - 2-D histogram
%          - Vector of X coordinate of center of X bins.
%          - Vector of Y coordinate of center of Y bins.
% Tested : Matlab 2011b
%     By : Eran O. Ofek                    Feb 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Xv=rand(100000,1).*2; Yv=rand(100000,1).*3+100; Mat=hist2d(Xv,Yv,[0 2],0.1,[100 103],0.1);
% Reliable: 2
%--------------------------------------------------------------------------

% reject points out of range
I    = find(Xv>=RangeX(1) & Xv<=RangeX(2) & Yv>=RangeY(1) & Yv<=RangeY(2));
Xv   = Xv(I);
Yv   = Yv(I);

NXv  = (Xv - RangeX(1))./StepX;
NYv  = (Yv - RangeY(1))./StepY;
VecX = (RangeX(1):StepX:(RangeX(2)-StepX)).' + StepX.*0.5;
VecY = (RangeY(1):StepY:(RangeY(2)-StepY)).' + StepY.*0.5;

XY   = NYv + floor(NXv).*numel(VecY); 
N    = histc(XY,(0:1:numel(VecX).*numel(VecY)).');
N    = N(1:end-1);

Mat  = reshape(N,length(VecY),length(VecX));

