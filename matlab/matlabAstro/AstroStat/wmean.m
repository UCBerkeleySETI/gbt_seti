function [M,E]=wmean(Vec,Err);
%------------------------------------------------------------------------------
% wmean function                                                     AstroStat
% Description: Calculated the weighted mean of a sample (ignoring NaNs).
% Input  : - Matrix containing an [Value, Error], or vector
%            containing [Value]. If a vector is given then get the
%            errors from the second argument.
%          - Optional column containing error on the values in the
%            first argument (alternatively, the errors may be
%            supplied in the second column of the first argument).
%            If Error supplied using both options, use the errors
%            from the second column of the first argument
% Output : - Wighted mean.
%	   - Weighted error on wighted mean.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                      June 1998
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [M,E]=wmean([1;2;3;4],[1;1;1;50]);
% Reliable: 2
%------------------------------------------------------------------------------
ColVal  = 1;
ColErr  = 2;
VecValue = Vec(:,ColVal);
if (size(Vec,2)==1),
   if (nargin==1),
      error('Errors must be supplied to wmean.m function');
   else
      VecError = Err;
   end
else
   VecValue = Vec(:,ColVal);
   VecError = Vec(:,ColErr);
end

% Ignore NaNs
I = find(isnan(VecValue)==0 & isnan(VecError)==0);

%E = sqrt(1./sum(1./VecValue(I).^2));
E = sqrt(1./sum(1./VecError(I).^2));
M = sum(VecValue(I)./(VecError(I).^2))./sum(1./(VecError(I).^2));
