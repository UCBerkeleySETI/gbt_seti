function [Mat,SimUM,Sim]=simcat_matchcoo(Sim,SimRef,varargin)
%--------------------------------------------------------------------------
% simcat_matchcoo function                                          ImPhot
% Description: Given a structure array (SIM) of images and source catalogs
%              and a reference image/catalog, match by coordinates the
%              sources in each image against the reference catalog.
%              For each requested property (e.g., 'XWIN_IMAGE') in the
%              catalog, the function returns a matrix of the property of
%              the matched sources. The matrix (ImageIndex,SourceIndex),
%              rows and columns corresponds to images, and sources
%              in the reference image, respectivelly.
%              In addition, the function returns a structure array
%              (element per image) of un-matched sources.
% Input  : - List of FITS images, structure aarray of images (SIM)
%            or any valid input to images2sim.m. The SIM input may also
%            contains a 'Cat' field. If not provided, then the Cat
%            field will be calculated.
%          - A single reference image. See image2sim.m for options.
%            If not provided, or empty, the reference image will be
%            selected from the list of images (1st argument) according
%            to the 'ChooseRef' option.
%            Default is [].
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Fields'- A cell array of field names ("properties") which
%                      will be returned as a matched matrix.
%                      Default is
%                      {'XWIN_IMAGE','YWIN_IMAGE','MAG_AUTO','MAGERR_AUTO'}.
%            'ColXc' - Column index or column name in the catalog
%                      corresponds to the X coordinates to use for matching
%                      Default is 'XWIN_IMAGE'. In general this can be
%                      an X coordinate or spherical longitude.
%            'ColYc' - Column index or column name in the catalog
%                      corresponds to the Y coordinates to use for matching
%                      Default is 'YWIN_IMAGE'. In general this can be
%                      an Y coordinate or spherical latitude.
%            'ColXr' - Like ColXc, but for the reference catalog.
%            'ColYc' - Like ColYc, but for the reference catalog.
%            'ChooseRef' - Method by which to choose the reference image
%                      from the images.
%                      'maxn' - image with max. number of sources (default).
%                      'first'- first image.
%                      'last' - last image.
%            'CooUnits' - The coordinates units:
%                      'pix' - pixel coordinates (default).
%                      'deg' - degrees (should be used with SExtractor
%                              outputs).
%                      'rad' - radians.
%            'SearchRad' - Search radius. Default is 2.
%                       If CooUnits='pix' then default is in pixels.
%                       If CooUnits='deg'/'rad' then default is in arcsec.
%            'CooType' - Force coordinates type to {'plane'|'sphere'}.
%                       If empty, then the value will be determinde
%                       according to 'CooUnits'.
%                       If CooUnits='pix' then CooType='plane'.
%                       If CooUnits='deg'/'rad' then CooType='sphere'.
%                       Default is empty.
%            'SearchCatPars' - A cell array of additional parameters to
%                       pass to search_cat.m. Default is {}.
%            'AddInfo' - Add additional information to the output matrix
%                       of matched sources (including, number of matched
%                       sources, JD, number of apperances and more).
%                       Default is true.
%            'Verbose' - Verbose. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, addcat2sim.m
% Output : - A structure array containing a field per requested property.
%            Each such field contains a matrix of matched sources
%            (ImageIndex,SourceIndex).
%            In addition, an optional 'Info' field containing the following
%            information:
%            .JD      - Vector of JD per image.
%            .NsrcRef - Number of sources in the reference image.
%            .Nim     - Number of images.
%            .IndRef  - Index of reference image (NaN if external).
%            .Nsrc    - Vector of number of sources found in each image.
%            .NsrcUM  - Vector of number of unmatched sources in each
%                       image (sources not matched to the reference).
%            .NsrcMatched - Number of sources matched in each image.
%            .Napp    - Vector of number of appearnces of each star
%                       in each image.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Mat,SimUM]=simcat_matchcoo(AlSim); % where AlSim are aligned images aligned using sim_align_shift.m
% Reliable: 2
%--------------------------------------------------------------------------
InvRAD = pi./180;


%ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
%CatColField     = 'Col';
CatColCellField = 'ColCell';
UserDataField   = 'UserData';

if (nargin==1),
    SimRef = [];
end

DefV.Fields           = {'XWIN_IMAGE','YWIN_IMAGE','MAG_AUTO','MAGERR_AUTO'};
DefV.ColXc            = 'XWIN_IMAGE';
DefV.ColYc            = 'YWIN_IMAGE';
DefV.ColXr            = 'XWIN_IMAGE';
DefV.ColYr            = 'YWIN_IMAGE';
DefV.ChooseRef        = 'maxn';
DefV.SearchRad        = 2;
DefV.CooUnits         = 'pix';
DefV.CooType          = [];  % {'plane'|'sphere'} or [].
DefV.SearchCatPars    = {};
DefV.AddInfo          = true;
DefV.Verbose          = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~iscell(InPar.Fields)),
    InPar.Fields = {InPar.Fields};
end

if (isempty(InPar.CooType)),
    switch lower(InPar.CooUnits),
        case 'pix'
            InPar.CooType   = 'plane';
        case {'deg','rad'}
            InPar.CooType   = 'sphere';
            InPar.SearchRad = InPar.SearchRad.*InvRAD./3600;
        otherwise
            error('Unknwon CooUnits option');
    end
end

% convert units to radians or pixels...
switch lower(InPar.CooUnits),
    case 'pix'
        UnitsConv = 1;
    case 'deg'
        UnitsConv = InvRAD;
    case 'rad'
        UnitsConv = 1;
    otherwise
        error('Unknwon CooUnits option');
end

%if (~isempty(strfind(InPar.ColXc,'ALPHA')) && strcmp(InPar.CooUnits,'pix')),
%    warning('CooUnits are pixels while Col is ALPHA');
%end

    

% read images and populate catalog
Sim = images2sim(Sim,varargin{:});
Sim = addcat2sim(Sim,varargin{:},'RePop',false);
Nim = numel(Sim);


% find column indices (convert column name to index)
[~,ColXc,ColYc,ColXr,ColYr] = col_name2ind(Sim(1).(CatColCellField),InPar.ColXc, InPar.ColYc,...
                                                                    InPar.ColXr, InPar.ColYr);



% Read reference image or use one of the images in SIM
if (isempty(SimRef)),
    switch lower(InPar.ChooseRef)
        case 'first'
            IndRef = 1;
        case 'last'
            IndRef = N;
        case 'maxn'
            Nsrc = zeros(Nim,1);
            for Iim=1:1:Nim,
                Nsrc(Iim) = size(Sim(Iim).(CatField),1);
            end
            [~,IndRef] = max(Nsrc);
        otherwise
            error('Unknown ChooseRef option');
    end
    SimRef = Sim(IndRef);
    
    if (InPar.Verbose),
        fprintf('Image number %d will be used as a reference image\n',IndRef);
    end
else      
    IndRef = NaN;
    SimRef = addcat2sim(SimRef,varargin{:},'RePop',false);
end

% sort ref catalog
SimRef.(CatField) = sortrows(SimRef.(CatField),ColYr);
NsrcRef           = size(SimRef.(CatField),1);

% initialize Mat
Nfields  = numel(InPar.Fields);
FieldInd = zeros(1,Nfields);
for Ifields=1:1:Nfields,
    Mat.(InPar.Fields{Ifields}) = nan(Nim,NsrcRef);
    % field index in Sim.Cat
    [~,FieldInd(Ifields)] = col_name2ind(Sim(1).(CatColCellField),InPar.Fields{Ifields});
end
% add additional information to Mat
if (InPar.AddInfo),
    Mat.Info.JD        = sim_julday(Sim);
    Mat.Info.NsrcRef   = NsrcRef;
    Mat.Info.Nim       = Nim;
    Mat.Info.IndRef    = IndRef;
    Mat.Info.Nsrc      = zeros(Nim,1);
    Mat.Info.NsrcUM    = zeros(Nim,1);
end


if (nargout>1),
    SimUM = struct(CatField,cell(Nim,1),UserDataField,cell(Nim,1));
end
% match all catalogs against reference
for Iim=1:1:Nim,
    % sort catalog
    
    Sim(Iim).(CatField) = sortrows(Sim(Iim).(CatField),ColYc);

    [ResS,CatUM] = search_cat(Sim(Iim).(CatField)(:,[ColXc,ColYc]).*UnitsConv,...
                              SimRef.(CatField)(:,[ColXr,ColYr]).*UnitsConv,[],...
                              InPar.SearchCatPars{:},...
                              'CooType',InPar.CooType,'SearchRad',InPar.SearchRad,'SearchMethod','binms1');
                
    % IcatWM are the indices of Ref stars which have a match in Cat
    IcatWM = find([ResS.Nfound]>0);
    % IrefM are the indices of matched stars in the Ref that corresponds to IcatWM
    IrefM  = [ResS(IcatWM).IndCat];
    
    for Ifields=1:1:Nfields,
        Mat.(InPar.Fields{Ifields})(Iim,IcatWM) = (Sim(Iim).(CatField)(IrefM,FieldInd(Ifields))).';
    end
    
    % add additional information to Mat
    if (InPar.AddInfo),
        Mat.Info.Nsrc(Iim)   = size(Sim(Iim).(CatField),1);
        Mat.Info.NsrcUM(Iim) = sum(CatUM); 
    end
    
    % store unmatched sources
    if (nargout>1),
        SimUM(Iim).(CatField)             = Sim(Iim).(CatField)(CatUM,:);
        SimUM(Iim).(UserDataField).FlagUM = CatUM; 
    end
    
end


% add additional information to Mat
% sources statistics
if (InPar.AddInfo),
    Mat.Info.NsrcMatched = sum(~isnan(Mat.(InPar.Fields{1})),2);
    Mat.Info.Napp        = sum(~isnan(Mat.(InPar.Fields{1})),1);
end
    