function [Mode,StD]=mode_fit(Array,varargin)
%--------------------------------------------------------------------------
% mode_fit function                                              AstroStat
% Description: Estimate the mode of an array by fitting a Gaussian to
%              the histogram of the array around its median.
%              Return also the Sigma of the Gaussian fit.
% Input  : - Array for which to calculate the mode and StD.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ElementPerBin' - Typical number of points per histogram bin.
%            'TrimEdge2D' - If the input is a 2D array, trim the edges
%                           by this number of pixels. Default is 5.
%            'Percent'    - The percentile of the data for which to fit
%                           the mode [lower upper]. This means that
%                           only values between the lower and upper
%                           percentiles will be used.
%                           Default is [0.025 0.9].
%            'Nbad'       - If histogram difference contains jumps
%                           which are larger than this value then these
%                           jumps will be removed from the fit.
% Output : - Mode.
%          - Fitted sigma.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Mode,StD]=mode_fit(randn(1000,1000))
% Reliable: 2
%--------------------------------------------------------------------------
SIGMA1 = 0.6827;


DefV.ElementPerBin     = 100;
DefV.TrimEdge2D        = 5;
DefV.Percent           = [0.025 0.9];
DefV.Rem0              = true;
DefV.Nbad              = 100;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (ndims(Array)==2 && InPar.TrimEdge2D>0),
    % Trim edges of 2D array to remove problems near image bounderies
    Array = Array(InPar.TrimEdge2D:1:end-InPar.TrimEdge2D,InPar.TrimEdge2D:1:end-InPar.TrimEdge2D);
end
    

Nel        = numel(Array);
Lower      = prctile(Array(:),InPar.Percent(1).*100);
Upper      = prctile(Array(:),InPar.Percent(2).*100);
%Percentile = err_cl(Array(:),InPar.Percent);

BinSize = (Upper-Lower).*InPar.ElementPerBin./Nel;
Edges   = (Lower-BinSize:BinSize:Upper+BinSize).';

if (InPar.Rem0),
    Array = Array(Array(:)~=0);
end

[N]=histc(Array(:),Edges);


%FlagGood = abs(N(2:end)-N(1:end-1))<InPar.Nbad;
%bar(Edges,N);
%Res=fit_gauss1d(Edges(FlagGood),N(FlagGood),1);
Res=fit_gauss1d(Edges,N,1);
%plot(medfilt1(Res.Resid,20))
%input('hi')
Mode = Res.X0;
StD  = Res.Sigma;
