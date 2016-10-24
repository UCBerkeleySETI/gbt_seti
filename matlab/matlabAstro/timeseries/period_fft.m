function PS=period_fft(Data,Normalization)
%--------------------------------------------------------------------------
% period_fft function                                           timeseries
% Description: Calculate power spectrum for evenly spaced time series
%              using fast Fourier transform.
%              See also period.m
% Input  : - Two columns matrix (e.g., [time, x]).
%          - Power spectrum normalization:
%            'no'    - no normalization.
%            'amp'   - amplitude normalization (e.g., sqrt of power).
%            'var'   - Normalize by variance, default.
% Output : - Power spectrum [Frequency, power].
% See also: period.m
% Tested : Matlab 7.6
%     By : Eran O. Ofek                    Jul 2009
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%--------------------------------------------------------------------------
Def.Normalization = 'var';
if (nargin==1),
   Normalization = Def.Normalization;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

DT = Data(2,1) - Data(1,1);  % Time diff. between measurments

Col.T = 1;
Col.Y = 2;
N = size(Data,1);
FFT = fft(Data(:,Col.Y));

Freq = ((1:1:N).'-1)./N;
Freq = Freq./DT;

PS  = [Freq, abs(FFT).^2];

switch lower(Normalization)
 case 'no'
    % no normalization
    % do nothing
 case 'amp'
    % amplitude normalization
    PS = [PS(:,1), sqrt(PS(:,2))./N];
 case 'var'
    % variance normalization
    Var = std(Data(:,Col.Y)).^2;
    PS = [PS(:,1), PS(:,2)./Var./N];
 otherwise
    error('Unknown Normalization option');
end
