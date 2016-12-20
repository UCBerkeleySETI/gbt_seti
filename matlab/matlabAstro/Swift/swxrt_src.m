function Out=swxrt_src(Path,RA,Dec,varargin);
%------------------------------------------------------------------------------
% swxrt_src function                                                     Swift
% Description: Given a Swift/XRT image and a position. Measure the counts
%              in the position and the bacground counts in an annulus
%              around the position.
% Input  : - A string containing a directory containing the XRT images.
%            The script will choose the image with the longest exposure
%            time. If empty matrix (i.e., []) then use current directory.
%            Alternatively this can be an event FITS file name.
%            If the directory contains additional directories, then the
%            program will assume each directory containing one XRT
%            observation and will reduce all the directories.
%            In this case the output argument will be a s tructure array.
%          - Object J2000.0 R.A. [radians]
%          - Object J2000.0 Dec. [radians]
%          * Arbitrary number of pairs of arguments ..., keyword,value,...
%            Possible keywords are:
%            'Aper'   - Aperture radius [arcsec]. Default is [7.2 50 100].
%                       [Aperture radius, Background annulus inner radius,
%                        Background annulus outer radius].
%            'Energy' - Energy range [low, high] in keV.
%                       Deafult is [0.2 10].
%            'HardCut'- Cutoff energy for calculation of hardness ratio.
%                       Default is 2 keV.
%            'Nsim'   - Number of simulations (default is 0). In each
%                       simulation the position of the aperture is perturbed
%                       randomally and the count rate in that aperture is
%                       measured.
%            'PertRad'- Maximal perturbation radius for simulations.
%                       Default is 5 [arcmin].
% Output : - Output structure containing the following fields:
%            .
% Tested : Matlab 7.13
%     By : Eran O. Ofek                    Jan 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Out=swxrt_src('./',225.000743./RAD, +1.881523./RAD,'Aper',[9 50 100])
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;

PWD = pwd;

if (isempty(Path)),
    Path = PWD;
end

DefV.Aper    = [7.2 50 100];
DefV.Energy  = [0.2 10];
DefV.HardCut = 2;
DefV.Nsim    = 0;
DefV.PertRad = 5; % [arcmin]
InPar = set_varargin_keyval(DefV,'n','def',varargin{:});

AreaRatio = InPar.Aper(1).^2./(InPar.Aper(3).^2 - InPar.Aper(2).^2);




RA  = convertdms(RA,'gH','r');
Dec = convertdms(Dec,'gD','R');

Multi = 0;
Ndir  = 1;
if (isdir(Path)),
   Dir = dir(Path);
   if (sum([Dir.isdir])>0),
      % contains directories - assume multiple observations of the same source
      Id = find([Dir.isdir]);
      Dir = Dir(Id);
      Dir = Dir(3:end);
      Ndir = length(Dir);
      Multi = 1;
      if (Ndir==0),
	     Ndir = 1;
         Multi = 0;
      end
   end
end


PathBase = Path;
for Idir=1:1:Ndir,
   switch Multi
    case 1
       Path = sprintf('%s%s%s%s%s%s%s',PathBase,Dir(Idir).name,filesep,'xrt',filesep,'event',filesep);
    otherwise
       % do nothing
   end


if (isdir(Path)),
   cd(Path);
   try
      gunzip('*.gz');
   end
   [~,List] = create_list('*_cl.evt',NaN);
else
   try
      gunzip('*.gz');
   end
   List{1} = Path;
end

%[KeywordVal,KeywordS] = mget_fits_keyword('*_cl.evt',{'EXPOSURE'});
[KeywordVal,KeywordS] = mget_fits_keyword(List,{'EXPOSURE'});
%KeywordS.EXPOSURE
[ExpTime,MaxInd]      = max([KeywordS.EXPOSURE]);
File = List{MaxInd};

[TelVal] = get_fits_keyword(File,{'TELESCOP'});
Telescope = lower(strtrim(TelVal{1}));
switch Telescope
 case 'swift'
    [KeywordVal] = get_fits_keyword(File,{'MJD-OBS'});
    JD = KeywordVal{1} + 2400000.5;
 case 'chandra'
    [KeywordVal] = get_fits_keyword(File,{'MJD_OBS'});
    JD = KeywordVal{1} + 2400000.5;
    [KeywordVal] = get_fits_keyword(File,{'TSTOP','TSTART'});
    ExpTime = KeywordVal{1} - KeywordVal{2};
 otherwise
    error('Unsupported telescope option');
end


[GoodInd,GoodTimes,ExpTime,Table,Col]=swxrt_filter_badtimes(File,[],'Nstd',4);
%Table       = fitsread(File,'BinTable');
%TableGT     = fitsread(File,'BinTable',2);
%GoodTimes   = sortrows([TableGT{1}, TableGT{2}],1);
%[~,~,~,Col] = get_fitstable_col(File,'BinTable');
%Ind         = find_ranges(Table{1},GoodTimes);
%[length(Ind), length(Table{1})]
%Table       = ind_cell(Table,Ind);
%
%if (length(Ind)~=length(Table{1})),
%   error('check GoodTimes');
%end

switch lower(Telescope)
 case 'swift'
    ColX = Col.X;
    ColY = Col.Y;
 case 'chandra'
    ColX = Col.x;
    ColY = Col.y;
 otherwise
    error('Unkwnon Telescope option');
end
[AllRA,AllDec] = swxrt_xy2coo(File,Table{ColX},Table{ColY},Telescope);

[SrcX,SrcY] = swxrt_coo2xy(File,RA,Dec,Telescope);

Dist = sphere_dist(RA,Dec,AllRA,AllDec);

TableX = Table{ColX};
TableY = Table{ColY};

switch Telescope
 case 'swift'
    TableEnergy = Table{Col.PI}./100;
    %TableT = Table{Col.Time};
    TableT = Table{Col.TIME};
 case 'chandra'
    TableEnergy = Table{Col.energy}./1000;
    TableT = Table{Col.time};

 otherwise
    error('Unsupported telescope option');
end
Isrc = find(Dist<InPar.Aper(1)./(RAD.*3600) & ...
            TableEnergy>InPar.Energy(1) & ...
            TableEnergy<InPar.Energy(2));
Ibck = find(Dist<InPar.Aper(3)./(RAD.*3600) & Dist>InPar.Aper(2)./(RAD.*3600) & ...
            TableEnergy>InPar.Energy(1) & ...
            TableEnergy<InPar.Energy(2));


%--- simulations ---
% look for count rate in different pointings
SimCounts = zeros(InPar.Nsim,1);
for Isim=1:1:InPar.Nsim,
   DistSim = sphere_dist(RA+(rand(1,1).*2-1).*InPar.PertRad./(60.*RAD)./cos(Dec),...
                         Dec+(rand(1,1).*2-1).*InPar.PertRad./(60.*RAD),...
                         AllRA,AllDec);
   IsrcSim = find(DistSim<InPar.Aper(1)./(RAD.*3600) & ...
                  TableEnergy>InPar.Energy(1) & ...
                  TableEnergy<InPar.Energy(2));
   SimCounts(Isim) = length(IsrcSim);
end
Out(Idir).SimCounts = SimCounts;


% find source center
switch Telescope
 case 'swift'
    Out(Idir).MeanX = mean(Table{Col.X}(Isrc));
    Out(Idir).MeanY = mean(Table{Col.Y}(Isrc));
 case 'chandra'
    Out(Idir).MeanX = mean(Table{Col.x}(Isrc));
    Out(Idir).MeanY = mean(Table{Col.y}(Isrc));
 otherwise
    error('Unsupported telescope option');
end


Out(Idir).File      = File;
Out(Idir).ExpTime   = ExpTime;
Out(Idir).JD        = JD;
Out(Idir).SrcCounts = length(Isrc);
Out(Idir).BckCounts = length(Ibck);
Out(Idir).SrcRate   = Out(Idir).SrcCounts ./Out(Idir).ExpTime;
Out(Idir).BckRate   = Out(Idir).BckCounts./Out(Idir).ExpTime  .* AreaRatio;
Out(Idir).NetSrcCounts = length(Isrc) - length(Ibck) .* AreaRatio;
Out(Idir).NetSrcRate   = Out(Idir).NetSrcCounts./Out(Idir).ExpTime;
Out(Idir).MeanRA    = mean(AllRA(Isrc));
Out(Idir).MeanDec   = mean(AllDec(Isrc));


% calculate the probability that the observed count rate is
% due to background noise
Out(Idir).FAB = fab_counts(sum(Out(Idir).SrcCounts),sum(Out(Idir).BckCounts.*AreaRatio));
%Out(Idir).UL1  = poissconf(sum(Out(Idir).SrcCounts),1-4.*(1-normcdf(1,0,1)));   % 1-sigma one sided upper limit
%Out(Idir).UL2  = poissconf(sum(Out(Idir).SrcCounts),1-4.*(1-normcdf(2,0,1)));   % 2-sigma one sided upper limit
%Out(Idir).UL3  = poissconf(sum(Out(Idir).SrcCounts),1-4.*(1-normcdf(3,0,1)));   % 3 sigma
Out(Idir).UL2  =counts_bck_ul(Out(Idir).SrcCounts,Out(Idir).BckCounts,AreaRatio,0.95);
Out(Idir).RA   = RA;
Out(Idir).Dec  = Dec;
Out(Idir).X    = SrcX;
Out(Idir).Y    = SrcY;
Out(Idir).SrcEnergyList = TableEnergy(Isrc);
Out(Idir).BckEnergyList = TableEnergy(Ibck);
Out(Idir).SrcXList      = TableX(Isrc);
Out(Idir).SrcYList      = TableY(Isrc);
Out(Idir).SrcTimeList   = TableT(Isrc);

% Calculate hardness radio
SrcNsoft    = sum(Out(Idir).SrcEnergyList<InPar.HardCut);
SrcNsoftErr = sqrt(SrcNsoft);
SrcNhard    = sum(Out(Idir).SrcEnergyList>=InPar.HardCut);
SrcNhardErr = sqrt(SrcNhard);
BckNsoft    = sum(Out(Idir).BckEnergyList<InPar.HardCut);
BckNsoftErr = sqrt(BckNsoft);
BckNhard    = sum(Out(Idir).BckEnergyList>=InPar.HardCut);
BckNhardErr = sqrt(BckNhard);

Out(Idir).SrcNsoft = SrcNsoft;
Out(Idir).SrcNhard = SrcNhard;
Out(Idir).BckNsoft = BckNsoft;
Out(Idir).BckNhard = BckNhard;
Out(Idir).MeanSrcEnergy    = mean(Out(Idir).SrcEnergyList);
Out(Idir).ErrMeanSrcEnergy = sqrt(Out(Idir).SrcEnergyList)./sqrt(length(Out(Idir).SrcEnergyList));

F_H    = inline('(a-b.*R)./(c-d.*R)','a','b','c','d','R');
% [ErrExp,ErrVar]=symerror('(a-b*R)/(c-d*R)','a','b','c','d') 
F_Herr = inline('(D_a.^2./(c - R.*d).^2 + (D_c.^2.*(a - R.*b).^2)./(c - R.*d).^4 + (D_b.^2.*R.^2)./(c - R.*d).^2 + (D_d.^2.*R.^2.*(a - R.*b).^2)./(c - R.*d).^4).^(1./2)','a','D_a','b','D_b','c','D_c','d','D_d','R');

Out(Idir).Hardness = F_H(SrcNhard,BckNhard,SrcNsoft,BckNsoft,AreaRatio);
Out(Idir).HardnessErr = F_Herr(SrcNhard,SrcNhardErr,BckNhard,BckNhardErr,SrcNsoft,SrcNsoftErr,BckNsoft,BckNsoftErr,AreaRatio);


cd(PWD);
      
end




