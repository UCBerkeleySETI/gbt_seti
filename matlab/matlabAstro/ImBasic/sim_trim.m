function Sim=sim_trim(Sim,varargin)
%--------------------------------------------------------------------------
% sim_trim function                                                ImBasic
% Description: Trim a set of images and save the result in a structure
%              array. If needed, then additional associated images
%              (e.g., mask) will be trimmed too.
% Input  : - Image to trim.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) Cell array of matrices.
%            (5) A file contains a list of image (e.g., '@list').
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'TrimSec' - Trim section. This can be either a vector of
%                        [xmin xmax ymin ymax] or a string containing
%                        the header keyword name containing the
%                        trim section. If empty then will use full image.
%                        Default is empty.
%            'TrimIm'  - Trim image field (ImageField) {true|false}.
%                        Default is true.
%            'TrimMask'- Trim mask image field (MaskField) {true|false}.
%                        Default is true.
%            'TrimErrIm'- Trim error image field (ErrImField) {true|false}.
%                        Default is true.
%            'TrimBack' - Trim background image field (BackImField) {true|false}.
%                        Default is true.
%            'TrimCat'  - Trim the catalog from sources outside the trim
%                        section, and transform the X/Y coordinates of the
%                        sources to the trimmed coordinate system
%                        {true|false}. Default is true.
%            'ColX'     - A cell array of X coordinate column names
%                        for which to transform the coordinates. The
%                        trimming will be done using the first element in
%                        the cell. Default is {'XWIN_IMAGE','X_IMAGE','X'}.
%            'ColY'     - A cell array of Y coordinate column names
%                        for which to transform the coordinates. The
%                        trimming will be done using the first element in
%                        the cell. Default is {'YWIN_IMAGE','Y_IMAGE','Y'}.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                         {true|false}. Default is true.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by images2sim.m.
% Output : - Structure array of trimmed images.
%            Note that header information (e.g., NAXIS1/2 and WCS) is not
%            modified.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: TrimSim=sim_trim(Sim,'TrimSec',[1 10 1 10]);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;


ImageField      = 'Im';
HeaderField     = 'Header';
%FileField       = 'ImageFileName';
MaskField       = 'Mask';
BackImField     = 'BackIm';
ErrImField      = 'ErrIm';
CatField        = 'Cat';
%CatColField     = 'Col';
CatColCellField = 'ColCell';


DefV.TrimSec      = [];
DefV.TrimIm       = true;
DefV.TrimMask     = true;
DefV.TrimErrIm    = true;
DefV.TrimBack     = true;
DefV.TrimCat      = true;
DefV.ColX         = {'XWIN_IMAGE','X_IMAGE','X'};
DefV.ColY         = {'YWIN_IMAGE','Y_IMAGE','Y'};
DefV.CopyHead     = 'y';
DefV.AddHead      = [];
DefV.DelDataSec   = true;
DefV.OutSIM       = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% prep the catalog trim columns
if (~iscell(InPar.ColX)),
    InPar.ColX = {InPar.ColX};
end
NcolX = numel(InPar.ColX);
if (~iscell(InPar.ColY)),
    InPar.ColY = {InPar.ColY};
end
NcolY = numel(InPar.ColY);


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

for Iim=1:1:Nim,
    
    CCDSEC = sim_ccdsec(Sim(Iim),InPar.TrimSec);
%     CCDSEC = get_ccdsec_head(Sim(Iim).(HeaderField),InPar.TrimSec);
%     if (any(isnan(CCDSEC))),
%          Size   = size(Sim(I).(ImageField));
%          CCDSEC = [1 Size(2) 1 Size(1)];
%     end

    % trim image
    if (isfield_notempty(Sim(Iim),ImageField) && InPar.TrimIm),
        Sim(Iim).(ImageField) = Sim(Iim).(ImageField)(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
    end
    
    % trim mask
    if (isfield_notempty(Sim(Iim),MaskField) && InPar.TrimMask),
        Sim(Iim).(MaskField) = Sim(Iim).(MaskField)(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
    end
    
    % trim error image
    if (isfield_notempty(Sim(Iim),ErrImField) && InPar.TrimErrIm),
        Sim(Iim).(ErrImField) = Sim(Iim).(ErrImField)(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
    end
    
    % trim background image
    if (isfield_notempty(Sim(Iim),BackImField) && InPar.TrimBack),
        Sim(Iim).(BackImField) = Sim(Iim).(BackImField)(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
    end
    
    % trim catalog data
    if (isfield_notempty(Sim(Iim),CatField) && InPar.TrimCat),
        ColIndX = col_name2indvec(Sim(Iim).(CatColCellField),InPar.ColX);
        ColIndY = col_name2indvec(Sim(Iim).(CatColCellField),InPar.ColY);
        FlagTrim = Sim(Iim).(CatField)(:,ColIndX(1))>=CCDSEC(1) & ...
                   Sim(Iim).(CatField)(:,ColIndX(1))<=CCDSEC(2) & ...
                   Sim(Iim).(CatField)(:,ColIndY(1))>=CCDSEC(3) & ...
                   Sim(Iim).(CatField)(:,ColIndY(1))<=CCDSEC(4);
          
        Sim(Iim).(CatField) = Sim(Iim).(CatField)(FlagTrim,:);
        
        Sim(Iim).(CatField)(:,ColIndX(~isnan(ColIndX))) = Sim(Iim).(CatField)(:,ColIndX(~isnan(ColIndX))) - CCDSEC(1)+1;
        Sim(Iim).(CatField)(:,ColIndY(~isnan(ColIndY))) = Sim(Iim).(CatField)(:,ColIndY(~isnan(ColIndY))) - CCDSEC(2)+1;
    end
        
    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'DelDataSec',InPar.DelDataSec,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('Trimmed section [%d %d %d %d]',CCDSEC)});
             
end

    