function [ck,sl]=cusum(x,L,K)
%---------------------------------------------------------------------------
% cusum function                                                 timeseries
% Description: cumulative sum (CUSUM) chart to detect non stationarity
%              in the mean of an evenly spaced series.
% Input  : - Matrix of evenly spaced data points [Index, Value].
%          - L (used to estimate the spectral density at zero frequency).
%            (L need to be small compared to N).
%            default is floor(0.1*N).
%          - K (used to estimate the spectral density at zero frequency).
%            default is 1.
% Output : - [k, cumulative sum up to k].
%          - The probability that the series is non-stationary.
% Reference : Koen & Lombard 1993 MNRAS 263, 287-308.
% Tested : Matlab 4.2
%     By : Eran O. Ofek                    Oct 1994
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: R=X.*0.001+rand(100,1); [ck,sl]=cusum([(1:1:100).',R]);
% Reliable: 2
%---------------------------------------------------------------------------
c_x = 1;
c_y = 2;
N = length(x(:,c_x));
if nargin==1,
   L = floor(0.1.*N);
   K = 1;
elseif nargin==2,
   K = 1;
elseif nargin==3,
   c_x = c_x;
else
   error('1, 2 or 3 args only');
end
y   = x(:,c_y) - mean(x(:,c_y));
ck = [x(:,c_x), cumsum(y)];
% calculating the spectral density at zero frequency.
f1 = K./N;
f2 = L./N;
df = f1:(f2-f1)./(L-K):f2;
p  = periodia(x,f1,f2,df);
S0 = sum(p(:,2))./L;
% normalizing the CUSUM.
ck = ck./sqrt(N.*S0);
Dn = max(abs(ck(:,2)));
sl = 2.*exp(-2.*Dn.*Dn);
