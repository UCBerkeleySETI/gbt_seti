function fdmt_test
%--------------------------------------------------------------------------
% fdmt_test function                                                 radio
% Description: Test the fdmt.m, fdmt_fft.m and fdmt_hybrid_coherent.m
%              algorithms.
% Input  : null
% Output : null
% See also: fdmt.m, fdmt_fft.m, fdmt_hybrid_coherent.m
% Tested : Matlab R2014a
%     By : Barak Zackay                    Nov 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%--------------------------------------------------------------------------

%% Timing the FDMT algorithm
Nt = 2^15;
Nf = 2^10;
Nd = 2^10;
test_FDMT_FFT = false;


if (test_FDMT_FFT == false),
    dataType = 'uint64';
    R = ones([Nt,Nf],dataType);
    tic;DM = fdmt(R,1200,1600,Nd,dataType);toc;
else
    dataType = 'double';
    R = ones([Nt,Nf],dataType);
    tic;DM = fdmt_fft(R,1200,1600,Nd,dataType);toc;
end

%% Testing the FDMT's hitting efficiency on an incoherent curve
N_freqs = 1024;
N_times = 16*1024;
f_min = 400;
f_max = 800;
N_DM = 1024;
N_show = 1024;
dataType = 'double';

Const1 = 1;
Const2 = 2;
Const3 = 3;
Thickness = 2;

test_FDMT_FFT = false;

[T,F] = meshgrid(1:1024*16,linspace(f_min,f_max,N_freqs));
%(This seems transposed (why T on left?) but it is correct because 
% meshgrid works in reverse) 

figure;
R0 = ones([N_freqs,N_times],dataType);
if (test_FDMT_FFT == false),
    RT0 = fdmt(R0',f_min,f_max,N_DM,dataType);
else 
    RT0 = fdmt_fft(R0',f_min,f_max,N_DM,dataType);
end
pcolor(double(RT0(1:N_show,1:N_show))); shading interp;
title('all ones as input, each pixel counts the amount of 1s added');



R1 = zeros([N_freqs,N_times],dataType);
S = T-900  + 4*40*1000000*((1/f_min)^2 - (1./(F).^2));
R1(S.^2 < (Thickness/2)^2) = Const1;
S = T-900  + 4*39*1000000*((1/f_min)^2 - (1./(F).^2));
R1(S.^2 < (Thickness/2)^2) = Const2;
S = T-900  + 4*41*1000000*((1/f_min)^2 - (1./(F).^2));
R1(S.^2 < (Thickness/2)^2) = Const3;

figure;
xaxis = linspace(0,1000 ,N_show);
yaxis = linspace(f_min,f_max,N_show);
pcolor(xaxis,yaxis,double(R1(1:N_show,1:N_show))); shading interp;
title('A mild DM, Input')
xlabel('Time [ms]');
ylabel('Frequency [MHz]')
figure;
if (test_FDMT_FFT == false),
    RT1 = fdmt(R1',f_min,f_max,N_DM,dataType);
else 
    RT1 = fdmt_fft(R1',f_min,f_max,N_DM,dataType);
end
pcolor(double(RT1(1:N_show,1:N_show))); shading interp;
title('A mild DM, Output')

hitting_efficiency = maxnd(RT1./RT0)/Const3


%% Testing FDMT on a coherently dispersed signal!
%
f_0 = 1200; %(in Mhz)
f_1 = 1600; %(in Mhz)
N_f = 2^10;
N_T = 2^10;
N_bins = 40;
dataType = 'double';
T_0 = 630000;
PulsePower = 0.2;
test_FDMT_FFT = false;
N_t_original = N_f*N_T*N_bins;
N_pulse_bins = N_bins*N_f*8;
DispersionConstant = 4.148808*10^3 *10^6; % Mhz * pc^-1 * cm^3
D = 70; % cm^-3 pc
practicalD = D*DispersionConstant;
N_DM = N_T;

f = linspace(0,f_1-f_0,N_t_original);
c = 0:N_t_original-1;
k = normrnd(0,PulsePower + 1,[1,N_pulse_bins]);
z = normrnd(0,1,[1,N_t_original]);

disp('Maximal theoretical SNR')
dE_max = sum(k(1:N_pulse_bins).^2) - N_pulse_bins*mean(z(N_pulse_bins+1:end).^2);
V_rand = var(z(N_pulse_bins+1:end).^2);
dE_max / (sqrt(V_rand * N_pulse_bins))

z(1:N_pulse_bins) = k(1:N_pulse_bins);
size(z);
size(f);
z = fft(z);
%s = exp(1i.*(2.*pi.*practicalD.*f.^2)./(f_0.^2.*(f_0+f))- 1i.*f*T_0);
%s = exp(1i.*(-2*pi .* (c - N_t_original/2)./N_t_original * practicalD./(f_0 + c.*(f_1-f_0)./N_t_original).^2) - 1i.*f*T_0);
s = exp(1i.*(2*pi * practicalD./(f_0 + f)) - 1i.*f*T_0);

size(s);
s = ifft(z.*s);
s = reshape(s,[N_f,N_bins,N_T]);
s = fft(s,[],1);
R2 = abs(s).^2;
R2 = reshape(sum(R2,2),[N_f,N_T]);
R2 = R2';
if test_FDMT_FFT == false,
    RT0 = fdmt(ones(size(R2)),f_0,f_1,N_DM,dataType);
else
    RT0 = fdmt_fft(ones(size(R2)),f_0,f_1,N_DM,dataType);
end
E = mean(R2(:));
V = var(R2(1:10)); %carefull not to put the pulse power in the variance calculation...
if test_FDMT_FFT == false,
    RT2 = fdmt(R2,f_0,f_1,N_DM,dataType);
else
    RT2 = fdmt_fft(R2,f_0,f_1,N_DM,dataType);
end
figure;
colormap('Gray');
xaxis = linspace(0,N_t_original/((f_1-f_0)*10^6)*1000,N_T); % *1000 for ms...
yaxis = linspace(f_0,f_1,N_f);
pcolor(xaxis,yaxis,double(R2')); shading flat;
hXLabel = xlabel('Time [ms]');
hYLabel = ylabel('Frequency [MHz]');
hTitle = title('Simulated Dispersion Signal, Input');
set([gca, hTitle, hXLabel, hYLabel], ...
    'FontSize'   , 16    , ...
    'FontName'   , 'MyriadPro-Regular');
figure;
% removing the expectancy
RT2 = RT2 - E.*RT0(1:size(RT2,1),1:size(RT2,2));
% removing the variance
RT2 = double(RT2)./sqrt(double(V.*RT0(1:size(RT2,1),1:size(RT2,2))));
disp('Acieved SNR')
max(RT2(:))


colormap('Hot');
xaxis = linspace(0,N_t_original/((f_1-f_0)*10^6)*1000,N_T);
yaxis = linspace(0,N_t_original/((f_1-f_0)*10^6)*1000,N_T);
pcolor(xaxis,yaxis,double(RT2(1:N_T,1:N_DM)')); shading flat;
hXLabel = xlabel('Time [ms]');
hYLabel =ylabel('\Deltat [ms]');
hTitle =title('Fast discrete Dispersion Measure Transform, Output');
set([gca, hTitle, hXLabel, hYLabel], ...
    'FontSize'   , 18    , ...
    'FontName'   , 'MyriadPro-Regular')

%%  HybridDedispersion_test 
SignalLength = 2^20;
PulseSig = 1.8;
PulseLength = 2^7;
f_min = 1200;
f_max = 1600;
d = 1; % The dispersion we are going to simulate and extract
maxSearchD = 1.5; % the maximal dispersion we are going to scan
NSigmas = 5;
PulsePos = 1000;

sig = normrnd(0,1,[1,SignalLength]);

sig(PulsePos:PulsePos + PulseLength - 1) = normrnd(0,PulseSig,[1,PulseLength]);
    
% Dispersion is just like dedispersion with a minus sign...
sig2 = CoherentDedispersion(sig,-d,f_min,f_max,false);
[d_0, H, DispersionAxis] = fdmt_hybrid_coherent(sig2, PulseLength, f_min, f_max, maxSearchD, NSigmas);
    
colormap('Hot');
xaxis = linspace(1,PulseLength,PulseLength);
yaxis = DispersionAxis;
pcolor(xaxis,yaxis,double(H(1:PulseLength,1:PulseLength)')); shading flat;
hXLabel = xlabel('Time [N_p * \tau]');
hYLabel =ylabel('DM [pc cm^-3]');
hTitle =title('Hybrid Coherent FDMT Algorithm, Output');
set([gca, hTitle, hXLabel, hYLabel], ...
    'FontSize'   , 18    , ...
    'FontName'   , 'MyriadPro-Regular')

