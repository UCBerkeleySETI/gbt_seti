function [GoodInd,GoodTimes,ExpTime,Table,Col]=swxrt_filter_badtimes(Input,GoodTimes,varargin)
%------------------------------------------------------------------------------
% swxrt_filter_badtimes function                                         Swift
% Description: Given an X-ray event file, look for time ranges in which
%              the background is elevated above the mean background rate
%              by a given amount.
% Input  : - A FITS binary table of the events.
%            Alternatively, this can be a vector of events.
%          - An optional GoodTimes maxtrix [start:end] times of good times
%            as obtained from the FITS binary table second extension.
%            If the first input is a FITS file then the GoodTimes are
%            retrieved from the FITS file and this parameter is ignored
%            (use empty matrix to ignore this parameter).
%          * Arbitrary number of pairs of input arguments:...,key,val,...
%            The following keywords ara available:
%            'NinBin'   - Tyipcal number of events in bin to use in the
%                         identification of bad times. Default is 20.
%            'Nstd'     - Number of StD above the median count rate which
%                         to reject as high background. Default is 4.
% Output : - Vector of indices of good events (i.e., events which are not
%            during bad times).
%          - Matrix of good time ranges, each row correspond to one range.
%            The first column for the start time and the second column for
%            the end time.
%          - Total exposure time of good times.
%          - If the first input argument is a FITS binary table, then
%            this is a cell array containing the columns in the FITS
%            binary table, but in which all the bad time events were
%            removed.
%            Otherwise this is an empty matrix.
%          - If the first input argument is a FITS binary table, then
%            this is a structure containing the columns information.
%            Otherwise this is an empty matrix.
% Tested : Matlab 7.13
%     By : Eran O. Ofek                       Feb 2012
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [GoodInd,GoodTimes,ExpTime,Table,Col]=swxrt_filter_badtimes(Input,[]);
% Reliable: 2
%------------------------------------------------------------------------------

if (nargin==1),
   GoodTimes = [];
end

DefV.NinBin = 20;
DefV.Nstd   = 4;
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

if (ischar(Input)),
    % load FITS binary table
    Table       = fitsread(Input,'BinTable');
    TableGT     = fitsread(Input,'BinTable',2);
    [~,~,~,Col] = get_fitstable_col(Input,'BinTable');
    GoodTimes   = [TableGT{1}, TableGT{2}];
    
    if (max(strcmp(fieldnames(Col),'TIME'))==1),
        EventTimes = Table{Col.TIME};
    elseif (max(strcmp(fieldnames(Col),'time'))==1),
        EventTimes = Table{Col.time};
    else
        error('could not find the time column in FITS binary file');
    end
else
    EventTimes = Input;
    GoodTimes  = GoodTimes;
    Col        = [];
    Table      = [];
    if (isempty(GoodTimes)),
        error('GoodTimes cannot be an empty matrix');
    end
end

% select good times as appear in the FITS binary table
Igt = find_ranges(EventTimes,GoodTimes);
EventTimes = EventTimes(Igt);

% for each range in GoodTimes calculate the median rate:
ExpTime = GoodTimes(:,2) - GoodTimes(:,1);  % exp time per range
Nrange = size(GoodTimes,1);
Nevt    = zeros(Nrange,1);
for Irange=1:1:Nrange,
    Iinr = find(EventTimes>GoodTimes(Irange,1) & EventTimes<GoodTimes(Irange,2));
    Nevt(Irange) = length(Iinr);
end
% remove GoodTimes ranges with no events
Iel0      = find(Nevt>0);
GoodTimes = GoodTimes(Iel0,:);
Nevt      = Nevt(Iel0);
ExpTime = GoodTimes(:,2) - GoodTimes(:,1);  % exp time per range
Nrange = size(GoodTimes,1);

% set the bin size such that there are ~NinBin counts per bin
CountsPerSec = median(Nevt./ExpTime);
BinSize      = InPar.NinBin./CountsPerSec;

% for each range in GoodTimes:
Iallbin = 0;
for Irange=1:1:Nrange,
    Iinr = find(EventTimes>GoodTimes(Irange,1) & EventTimes<=GoodTimes(Irange,2));
    Nbin = ceil(ExpTime(Irange)./BinSize);   % integral number of bins such that each bin contains ~NinBin counts
    ActualBinSize = (GoodTimes(Irange,2) - GoodTimes(Irange,1))./Nbin;
    Edges = (GoodTimes(Irange,1):ActualBinSize:GoodTimes(Irange,2))';
    N = histc(EventTimes(Iinr),Edges);
    N = N(1:end-1);
  
    All(Irange).Xs = Edges(1:end-1).';
    All(Irange).Xe = Edges(2:end).';
    All(Irange).N = N.';
    % find the indices of events in each bin
    for Ibin=1:1:Nbin,
        Iallbin = Iallbin + 1;
        AllBin(Iallbin).Ind = find(EventTimes>Edges(Ibin) & EventTimes<=Edges(Ibin+1)).';
        AllBin(Iallbin).Xs  = Edges(Ibin);
        AllBin(Iallbin).Xe  = Edges(Ibin+1);
    end
end
Nallbin = Iallbin;

Bins.Xs    = [All.Xs].';                % bin start time
Bins.Xe    = [All.Xe].';                % bin end time
Bins.DT    = Bins.Xe - Bins.Xs;         % exposure time per bin
Bins.T     = 0.5.*(Bins.Xe + Bins.Xs);  % bin mid time
Bins.N     = [All.N].';                 % number of events in bin
Bins.ErrN  = sqrt(Bins.N);              % error ~ sqrt(N)
Bins.CR    = Bins.N./Bins.DT;           % count rate in bin
MedianDT   = median(Bins.DT);           % median exposure time in bins

Bins.NormN = Bins.N .* MedianDT./Bins.DT;     % counts in standard bin of lengthe MedianDT
Bins.NormNe= Bins.ErrN .* MedianDT./Bins.DT;  % normalized error

% search for count rates which are within Nstd-sigma from median count rate
Flag    = (Bins.NormN - Bins.NormNe.*InPar.Nstd) < median(Bins.NormN);
FlagInd = find(Flag);
GoodInd = [AllBin(Flag).Ind].';    % all indices in good time bins

Xs = [AllBin(Flag).Xs].';
Xe = [AllBin(Flag).Xe].';
FlagGaps = [(Xs(2:end)~=Xe(1:end-1)); 0];
FlagGapsS = [1==1; Xs(2:end)~=Xe(1:end-1)];
FlagGapsE = [Xe(1:end-1)~=Xs(2:end); 1==1];
%[Xs, Xe, FlagGapsS, FlagGapsE]

GoodTimes = [Xs(FlagGapsS), Xe(FlagGapsE)];
ExpTime   = sum(GoodTimes(:,2) - GoodTimes(:,1));


if (nargout>3),
   if (ischar(Input)),
       Table = ind_cell(Table,GoodInd);
   end
end
       
