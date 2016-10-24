function Sim=xcat2sim(Sim,varargin)
%--------------------------------------------------------------------------
% xcat2sim function                                                ImBasic
% Description: cross-correlate external astronomical catalogs with
%              a catalog or SIM catalog.
% Input  : - Images or SIM images or SIM catalog in any format acceptable
%            by addcat2sim.m and images2sim.m.
%            The 'Cat' field in the SIM will be cross-correlated with
%            external catalogs.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ExtCats' - Cell arry of external catalogs to match against.
%                        Each entry is an external catalog mat file name
%                        or function_handle.
%                        Default is
%                        {@wget_sdss,'FIRST','NVSS','PGC','ROSAT_BSC','ROSAT_faint'}.
%            'Dictionary'- External catalog dictionary. This is a 3 column
%                        cell array: {catalogName, SearchRadius[rad],
%                        ExtraParameters}.
%            'MatchOutType'- Indicating what type of output to return:
%                        'det_num' - add a flag with number of matches per
%                                    source
%                        'add_allcols'-add all columns of matched catalog.
%                        'add_cols' - add selected columns of matched cat.
%            'GetColumns' - Two columns cell array indicating which columns
%                         to return per catalog for the 'add_cols' option.
%            'RePop'    - Re populate SIM catalog. Default is false.
%            'ColRA'    - Column name (or index) of J2000.0 RA in the SIM catalog
%                         to use for matching. Default is 'ALPHAWIN_J2000'.
%            'ColDec'   - Column name (or index) of J2000.0 Dec in the SIM catalog
%                         to use for matching. Default is 'DELTAWIN_J2000'.
%            'IsRefdeg' - Indicating the units of the RA/Dec in the SIM
%                         catalog. true means deg, false is for radians.
%                         Default is true (deg).
%            'Verbose'  - Verbose message printing. Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            addcat2sim.m, images2sim.m, image2sim.m
% Output : - The SIM images and catalogs with the cross-correlation
%            Information.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SimC=xcat2sim(Sim);
% Reliable: 2
%--------------------------------------------------------------------------
InvRAD      = pi./180;
RAD         = 180./pi;
ARCSEC_DEG  = 3600;


%ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';
%UserDataField   = 'UserData';


DefV.ExtCats          = {@wget_sdss,'APASS_htm','FIRST','NVSS','PGC','ROSAT_BSC','ROSAT_faint'};
DefV.Dictionary       = {'wget_sdss',     1.5.*InvRAD./ARCSEC_DEG, {};...
                         'wget_usnob1',   1.5.*InvRAD./ARCSEC_DEG, {};...
                         'wget_ucac4',    1.5.*InvRAD./ARCSEC_DEG, {};...
                         'wget_2mass',    1.5.*InvRAD./ARCSEC_DEG, {};...
                         'APASS_htm'      1.5.*InvRAD./ARCSEC_DEG, {};...
                         'FIRST',         2.0.*InvRAD./ARCSEC_DEG, {};...
                         'FIRST_htm',     2.0.*InvRAD./ARCSEC_DEG, {};...
                         'NVSS',          15.0.*InvRAD./ARCSEC_DEG, {};...
                         'NVSS_htm',      15.0.*InvRAD./ARCSEC_DEG, {};...
                         'PGC',           10.0.*InvRAD./ARCSEC_DEG, {};...
                         'IRAS'           30.0.*InvRAD./ARCSEC_DEG, {};...
                         'ROSAT_BSC',     30.0.*InvRAD./ARCSEC_DEG, {};...
                         'ROSAT_faint',   45.0.*InvRAD./ARCSEC_DEG, {};...
                         'Abell',         300.*InvRAD./ARCSEC_DEG,  {};...
                         'MaxBCG',        300.*InvRAD./ARCSEC_DEG,  {};...
                         'NVSS_RM',       15.0.*InvRAD./ARCSEC_DEG, {};...
                         'Pulsar',        60.0.*InvRAD./ARCSEC_DEG, {};...
                         'PhotCatPTF',    1.5.*InvRAD./ARCSEC_DEG, {};...
                         'SDSS_Clusters', 300.*InvRAD./ARCSEC_DEG,  {};...
                         'Landolt',       1.5.*InvRAD./ARCSEC_DEG, {};...
                         'Tycho2JHKgriz', 2.*InvRAD./ARCSEC_DEG, {};...
                         'Tycho2',        2.*InvRAD./ARCSEC_DEG, {};};                         
DefV.MatchOutType     = {'det_num','add_cols'};    % 'det_num','add_allcols','add_cols'
DefV.GetColumns       = {'wget_sdss',{'ra','dec','type','modelMag_u','modelMagErr_u','modelMag_g','modelMagErr_g','modelMag_r','modelMagErr_r','modelMag_i','modelMagErr_i','modelMag_z','modelMagErr_z'};...
                         'wget_usnob1',{'RA','Dec','B2mag','R2mag'};...
                         'wget_ucac4',{'RA','Dec'};...
                         'wget_2mass',{'RA','Dec','J','H','K'};...
                         'APASS_htm',{'V','B','g','r','i','Verr','Berr','gerr','rerr','ierr'};...
                         'FIRST',{'FPEAK'};...
                         'FIRST_htm',{'FPEAK'};...
                         'NVSS',{'Flux'};...
                         'NVSS_htm',{'Flux'};...
                         'PGC',{};...
                         'IRAS',{'Flux12','Flux25','Flux60','Flux100'};...
                         'ROSAT_BSC',{'SrcCountRate'};...
                         'ROSAT_faint',{'CountRate'};...
                         'Abell',{'z'};...
                         'MaxBCG',{'photo_z'};...
                         'NVSS_RM',{};...
                         'Pulsar',{};...
                         'PhotCatPTF',{'R_PTF','errR_PTF'};...
                         'SDSS_Clusters',{'z_photo'};...
                         'Landolt',{'V','BV','UB','VR','RI','VI','errV','errBV','errUB','errVR','errRI','errVI'};...
                         'Tycho2JHKgriz',{};...
                         'Tycho2',{'BT','VT'}};
                         
DefV.RePop            = false;
DefV.ColRA            = 'ALPHAWIN_J2000';  % column name or index
DefV.ColDec           = 'DELTAWIN_J2000';
DefV.IsRefdeg         = true;  % assume SExtractor catalog provides deg.
DefV.Verbose          = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~iscell(InPar.MatchOutType)),
    InPar.MatchOutType = {InPar.MatchOutType};
end
Nouttype = numel(InPar.MatchOutType);


% read image and catalog into SIM
Sim  = addcat2sim(Sim,varargin{:},'RePop',InPar.RePop);
Nsim = numel(Sim);

% get SIM footprints
Foot = sim_footprint(Sim,varargin{:});

% FFU - may add in future version
% check if all SIMs have the same/similar footprints
% [Foot.GcebLong]
% FootAll.GcenLong = 
% FootAll.GcenLat  = 
% FootAll.Radius   = 
HaveSimilarFoot = false;


[~,ColRA, ColDec] = col_name2ind(Sim(1).(CatColCellField),InPar.ColRA,InPar.ColDec);

Nec = numel(InPar.ExtCats);  % number of external catalos to cross-correlate with

MatchRad = zeros(Nec,1);
for Icat=1:1:Nec,
    % for each external catalog
    CatalogName = InPar.ExtCats{Icat};
    if (isa(CatalogName,'function_handle')),
        CatalogNameStr = func2str(CatalogName);
    else
        CatalogNameStr = CatalogName;
    end
    
    % catalog search radius from dictionary
    Idic = find(strcmpi(InPar.Dictionary(:,1),CatalogNameStr));
    MatchRad(Icat) = InPar.Dictionary{Idic,2};
    ExtraCatPar    = InPar.Dictionary{Idic,3};
        
    if (InPar.Verbose),
        if (isa(CatalogName,'function_handle')),
            CatalogNameStr = func2str(CatalogName);
        else
            CatalogNameStr = CatalogName;
        end
        fprintf('Cross-match %d SIM catalogs with catalog: %s and %7.2f arcsec search radius\n',Nsim,CatalogNameStr,MatchRad(Icat).*RAD.*ARCSEC_DEG);
    end
    
    for Isim=1:1:Nsim,
        % for each SIM image/catalog

        if (HaveSimilarFoot),
            if (Isim==1),
                % run search only for first image
                [Cat,ColCell,Col] = get_cat(CatalogName,FootAll.GcenLong,FootAll.GcenLat,FootAll.Radius,ExtraCatPar{:});
            end
        else
            % run search for all images
            [Cat,ColCell,Col] = get_cat(CatalogName,Foot(Isim).GcenLong,Foot(Isim).GcenLat,Foot(Isim).Radius,ExtraCatPar{:});
        end
        
        % match with SIM catalog
        Sim(Isim).(CatField) = sortrows(Sim(Isim).(CatField),ColDec);
        % check if catalog is sorted
        if (~issorted(Cat(:,Col.Dec))),
            error('Catralog is not sorted by declination');
        end
        [Res,CatUM]=search_cat(Cat(:,[Col.RA, Col.Dec]),...
                               Sim(Isim).(CatField)(:,[ColRA,ColDec]), [],...
                               'SearchRad',MatchRad(Icat),...
                               'SearchMethod','binms',...
                               'CooType','sphere',...
                               'IsRefdeg',InPar.IsRefdeg,...
                               'IsRad',true);
        
        % prepare output
        for Iouttype=1:1:Nouttype,
            switch lower(InPar.MatchOutType{Iouttype})
                case 'det_num'
                    ColumnName = sprintf('%s_Nmatch',CatalogNameStr);
                    
                    Ncol = size(Sim(Isim).(CatField),2) + 1;
                    Sim(Isim).(CatField)(:,Ncol)         = [Res.Nfound].';  % add column of number of matched per source
                    Sim(Isim).(CatColField).(ColumnName) = Ncol;
                    Sim(Isim).(CatColCellField){Ncol}    = ColumnName;
                    
                case 'add_allcols'
                    Ncol    = size(Sim(Isim).(CatField),2);
                    NcolExC = size(Cat,2);
                    VecCol  = (Ncol+1:1:Ncol+NcolExC);
                    Sim(Isim).(CatField)(:,VecCol) = NaN;
                    ColCellCat = strcat(sprintf('%s_',CatalogNameStr),ColCell);
                    Sim(Isim).(CatColCellField) = [Sim(Isim).(CatColCellField), ColCellCat];
                    Sim(Isim).(CatColField)     = cell2struct(num2cell(1:1:Ncol+NcolExC),Sim(Isim).(CatColCellField),2);
                    for Ires=1:1:numel(Res),
                        if (~isempty(Res(Ires).IndCat)),
                            % add Cat row
                            Sim(Isim).(CatField)(Ires,VecCol) = Cat(Res(Ires).IndCat(1),:);
                        end
                    end
                    
                case 'add_cols'
                    
                    Icol = find(strcmpi(InPar.GetColumns(:,1),CatalogNameStr));
                    if (isempty(Icol)),
                        % no instructions in GetColumns
                        % default is to present all columns
                        Ncol    = size(Sim(Isim).(CatField),2);
                        NcolExC = size(Cat,2);
                        VecCol  = (Ncol+1:1:Ncol+NcolExC);
                        Sim(Isim).(CatField)(:,VecCol) = NaN;
                        ColCellCat = strcat(sprintf('%s_',CatalogNameStr),ColCell);
                        Sim(Isim).(CatColCellField) = [Sim(Isim).(CatColCellField), ColCellCat];
                        Sim(Isim).(CatColField)     = cell2struct(num2cell(1:1:Ncol+NcolExC),Sim(Isim).(CatColCellField),2);
                        for Ires=1:1:numel(Res),
                            if (~isempty(Res(Ires).IndCat)),
                                % add Cat row
                                Sim(Isim).(CatField)(Ires,VecCol) = Cat(Res(Ires).IndCat(1),:);
                            end
                        end    
                    else
                        % present only columns specified in GetColumns
                        ColToAdd = InPar.GetColumns{Icol,2};
                        ColInd   = col_name2indvec(ColCell,ColToAdd);
                        
                        Ncol    = size(Sim(Isim).(CatField),2);
                        NcolExC = numel(ColInd);
                        VecCol  = (Ncol+1:1:Ncol+NcolExC);
                        Sim(Isim).(CatField)(:,VecCol) = NaN;
                        ColCellCat = strcat(sprintf('%s_',CatalogNameStr),ColToAdd);
                        Sim(Isim).(CatColCellField) = [Sim(Isim).(CatColCellField), ColCellCat];
                        Sim(Isim).(CatColField)     = cell2struct(num2cell(1:1:Ncol+NcolExC),Sim(Isim).(CatColCellField),2);
                        for Ires=1:1:numel(Res),
                            if (~isempty(Res(Ires).IndCat)),
                                % add Cat row
                                Sim(Isim).(CatField)(Ires,VecCol) = Cat(Res(Ires).IndCat(1),ColInd);
                            end
                        end    
                    end
                    
                otherwise
                    error('Unknown MatchOutType option');
            end
        end
        
    end
end


