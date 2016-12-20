function [Corr,Len]=ccf_diff(List1,List2,varargin)
%--------------------------------------------------------------------------
% ccf_diff function                                             timeseries
% Description: Given two equally sapced time series of the same length,
%              calculate the difference between points of distance N
%              and calculate the correlation between this differences
%              in the two serieses. Return the correlation as a function
%              of distance N. This is usefull if trying to estimate if
%              the correlation between two series is due to high or low
%              frequency. Supports partial correlations.
% Input  : - A column vector of the first equally spaced time series.
%          - A column vector of the second equally spaced time series.
%            Must have the same length as the first series.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'DiffVec' - Vector of differences in which to calculate
%                        correlation. Default is (1:1:100).'.
%            'Type'    - Correlation type. See corr.m for options.
%                        Default is 'Spearman'.
%            'Part1'   - Additional columns corresponding to the first
%                        series which to fix for partial correlations.
%                        Default is empty.
%            'Part2'   - Additional columns corresponding to the second
%                        series which to fix for partial correlations.
%                        Default is empty.
% Output : - Correlation as a function of difference.
%          - Length of series used to calculate each correlation.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Apr 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: List1=rand(10000,1); List2=rand(10000,1);
%          [Corr,Len]=ccf_diff(List1,List2);
%          plot(Corr); hold on; plot(1./sqrt(Len),'r-')
% Reliable: 2
%--------------------------------------------------------------------------

Col.F = 1;

DefV.DiffVec = (1:1:100).'; 
DefV.Type    = 'Spearman';
DefV.Part1   = [];
DefV.Part2   = [];
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


N1 = size(List1,1);
N2 = N1; %size(List2,1);  % must equal N1

Nd = numel(InPar.DiffVec);
Corr = zeros(Nd,1);
Len  = zeros(Nd,1);
for Id=1:1:Nd,
    DiffN = InPar.DiffVec(Id);
    
    D1 = [];
    D2 = [];
    for Idn=1:1:DiffN,
        Ip = (DiffN+1:DiffN:N1) - (Idn-1);
        Im = Ip-DiffN;
        FlagGood = Ip>0 & Im>0 & Ip<N1 & Im<N2;
        
        D1 = [D1; [List1(Ip(FlagGood),Col.F) - List1(Im(FlagGood),Col.F)]];
        D2 = [D2; [List2(Ip(FlagGood),Col.F) - List2(Im(FlagGood),Col.F)]];
    end
        
    Len(Id) = length(D1);
    if (isempty(InPar.Part1) || isempty(InPar.Part2)),
        Tmp = corr([D1, D2],'type',InPar.Type);
        Corr(Id) = Tmp(1,2);
    else
        % partial correlations
        Npart = size(InPar.Part1,2);
        PartD1 = zeros(length(D1),Npart);
        PartD2 = zeros(length(D2),Npart);
        for Ipart=1:1:Npart,
        
           PartD1(:,Ipart) = InPar.Part1(DiffN+1:DiffN:end,Col.F) - InPar.Part1(1:DiffN:end-DiffN);
           PartD2(:,Ipart) = InPar.Part2(DiffN+1:DiffN:end,Col.F) - InPar.Part2(1:DiffN:end-DiffN);
           Tmp = partialcorr([D1, D2, PartD1, PartD2],'type',InPar.Type);
           Corr(Id) = Tmp(1,2);
        end
    end
end    

