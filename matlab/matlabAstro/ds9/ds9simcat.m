function Sim=ds9simcat(Sim,varargin)
%--------------------------------------------------------------------------
% ds9simcat function                                                   ds9
% Description: Display SIM images (or any other images) in ds9, create
%              or use associate source catalog, query the source catalog
%              and display markers around selected sources.
% Input  : - SIM, FITS or any other types of images to display.
%            See images2sim.m for image type options.
%            If the SIM include a 'Cat' entry then it will be used,
%            otherwise the 'Cat' entry will be populated using
%            addcat2sim.m (unless 'RePop' is false).
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'DispImage'- Display image into ds9 {true|false}. If false
%                       will assume that the images are already in ds9.
%                       Default is true.
%            'Frame'  - Vector of frame indices in which to load the
%                       images. If empty will use frame 1:NumberOfImages.
%                       Default is empty.
%            'StartDS9'- Start ds9 {true|false}. Default is true.
%            'DeleteRegions'- Delete existing regions/markers from
%                       image. Default is false.
%            'RePop'  - Re populate 'Cat' field if exist.
%                       Default is false.
%            'QueryString' - A cell array of query strings. Each query
%                       string will be executed for each image and
%                       markers will be plotted for the selected objects.
%                       The entry number in the query string cell
%                       corresponds to the entry number in the markers
%                       parameters cell arrays. If a query string contains
%                       true, then all sources will be marked.
%                       Default is {true}.
%            'Coo'    - Cell arry of coordinate type corresponding to
%                       each query string. If single element then identical
%                       for all query strings. This can be {'image'|'fk5'}.
%                       Default is 'image'.
%            'ColX'   - Cell arry of X coordinate column name or index
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {'XWIN_IMAGE'}.
%            'ColY'   - Cell arry of Y coordinate column name or index
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {'YWIN_IMAGE'}.
%            'Type'   - Cell arry of marker symbol types
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {'circle'}.
%                       See ds9_plotregion.m for options.
%            'Color'  - Cell arry of marker symbol colors
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {'red'}.
%                       See ds9_plotregion.m for options.
%            'Width'  - Cell arry of marker symbol line widths
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {1}.
%                       See ds9_plotregion.m for options.
%            'Size'  - Cell arry of marker symbol size
%                       corresponding to each query string. If single
%                       element then identical for all query strings.
%                       Default is {20}.
%                       See ds9_plotregion.m for options.
%            'Verbose' - SHow verbose messages. Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            image2sim.m, images2sim.m, ds9_disp.m
% Output : - Sim images.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9simcat('*.fits');  % display all images, run sextractor and mark all sources
%          ds9simcat('*.fits','QueryString',{'XWIN_IMAGE>500','XWIN_IMAGE<500'},'Color',{'red','white'});
%          Sim1=xcat2sim('A.fits','verbose',true);
%          ds9simcat(Sim1,'QueryString',{'(wget_sdss_modelMag_g-wget_sdss_modelMag_r)<0.3 & wget_sdss_modelMag_i<19'});
%          ds9simcat(Sim1,'QueryString',{'(wget_sdss_modelMag_g-wget_sdss_modelMag_r)<0.3 & wget_sdss_modelMag_i<19','NVSS_Nmatch>0'},'Color',{'red','blue'}); 
% Reliable: 2
%--------------------------------------------------------------------------


ImageField      = 'Im';
%HeaderField     = 'Header';
%FileField       = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
%CatColField     = 'Col';
CatColCellField = 'ColCell';


DefV.DispImage        = true;  % if fakse will assume image is already loaded to ds9
DefV.Frame            = [];
DefV.StartDS9         = true;
DefV.DeleteRegions    = false;
DefV.RePop            = false;   % repopulate catalog
DefV.QueryString      = {true};
DefV.Coo              = {'image'};  % 'iamge' | 'fk5'
DefV.ColX             = {'XWIN_IMAGE'};  % or index
DefV.ColY             = {'YWIN_IMAGE'};  % or index
DefV.Type             = {'circle'};
DefV.Color            = {'red'};
DefV.Width            = {1};
DefV.Size             = {20};
DefV.Verbose          = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% FFN = FITS file names
[Sim,FFN] = images2sim(Sim,varargin{:});
Nim       = numel(Sim);
Sim       = addcat2sim(Sim,varargin{:},'RePop',InPar.RePop);

if (isempty(InPar.Frame)),
    InPar.Frame = (1:1:Nim);
end

% populate in cells and count
if (~iscell(InPar.QueryString)),
    InPar.QueryString = {InPar.QueryString};
end
Nqs = numel(InPar.QueryString);

if (~iscell(InPar.Coo)),
    InPar.Coo = {InPar.Coo};
end
Ncoo = numel(InPar.Coo);

if (~iscell(InPar.ColX)),
    InPar.ColX = {InPar.ColX};
end
Ncolx = numel(InPar.ColX);

if (~iscell(InPar.ColY)),
    InPar.ColY = {InPar.ColY};
end
Ncoly = numel(InPar.ColY);

if (~iscell(InPar.Type)),
    InPar.Type = {InPar.Type};
end
Ntype = numel(InPar.Type);

if (~iscell(InPar.Color)),
    InPar.Color = {InPar.Color};
end
Ncolor = numel(InPar.Color);

if (~iscell(InPar.Width)),
    InPar.Width = {InPar.Width};
end
Nwidth = numel(InPar.Width);

if (~iscell(InPar.Size)),
    InPar.Size = {InPar.Size};
end;
Nsize = numel(InPar.Size);

% start ds9
if (InPar.StartDS9),
    ds9_start;
end

% go over all images
for Iim=1:1:Nim,
    if (InPar.DispImage),
        % display image
        if (isempty(FFN{Iim})),
            % display SIM
            %ds9_disp(Sim(Iim).(ImageField),InPar.Frame(Iim),varargin{:},'StartDS9',false);
            ds9_disp(Sim(Iim),InPar.Frame(Iim),varargin{:},'StartDS9',false);
        else
            % display FITS (faster)
            ds9_disp(FFN{Iim},InPar.Frame(Iim),varargin{:},'StartDS9',false);
        end
    end
    
    % delete previous regions
    if (InPar.DeleteRegions),
        ds9_regions('delete');
    end
    
    % go over all query strings
    for Iqs=1:1:Nqs,
        % indices of marker properties in cell
        Itype   = min(Iqs,Ntype);
        Icolor  = min(Iqs,Ncolor);
        Iwidth  = min(Iqs,Nwidth);
        Isize   = min(Iqs,Nsize);
        Icolx   = min(Iqs,Ncolx);
        Icoly   = min(Iqs,Ncoly);
        Icoo    = min(Iqs,Ncoo);
        
        % Get coordinate column indices
        [~,ColX,ColY] = col_name2ind(Sim(Iim).(CatColCellField),InPar.ColX{Icolx},InPar.ColY{Icoly});
        
        % query catalog using query string
        Flag = query_columns(Sim(Iim).(CatField),Sim(Iim).(CatColCellField),InPar.QueryString{Iqs});
        
        if (InPar.Verbose),
            fprintf('Image number %d and query string %d\n',Iim,Iqs);
            fprintf('      Mark %d sources\n',numel(find(Flag)));
        end
        
        % plot markers
        ds9_plotregion(Sim(Iim).(CatField)(Flag,ColX),Sim(Iim).(CatField)(Flag,ColY),...
                       'Append','n',...
                       'Coo',InPar.Coo{Icoo},...
                       'Type',InPar.Type{Itype},...
                       'Size',InPar.Size{Isize},...
                       'Color',InPar.Color{Icolor},...
                       'Width',InPar.Width{Iwidth});
    end
end

            
        
        