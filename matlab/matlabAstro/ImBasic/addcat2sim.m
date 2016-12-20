function Sim=addcat2sim(Sim,varargin)
%--------------------------------------------------------------------------
% addcat2sim function                                              ImBasic
% Description: Add a catalog into structure image array (SIM) or upload
%              images and catalogs into SIM. The catalogs can be provided
%              as a FITS table, or extracted from images.
% Input  : - A structure image array, SIM, a list of FITS images and more
%            (see images2sim.m for options).
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'RePop'     - Re-populate catalog even if it exist
%                          {true|false}. Default is true.
%            'UseSimImageName' - Use FITS image name from SIM File field.
%                          Default is false.
%            'ColSort'   - Sort output catalog by a specific column.
%                          This is either empty (no sorting), scalar
%                          (column index) or string (column name).
%                          Default is 'YWIN_IMAGE'.
%            'ImageName' - A cell array list of image names from which to
%                          extract sources. Default is {}.
%                          See create_list.m for more options.
%                          If provided, this will override the FITStable
%                          and file name options.
%            'FITStable' - A cell array list of FITS table names from which
%                          to read the sources. Default is {}.
%                          See create_list.m for more options.
%                          If provided, this will override the file name
%                          option. If both ImageName and FITStable are
%                          not provided then will read the image name
%                          from the SIM, and attempt to extract sources.
%            'ProgName' - Function handle to use for source extraction.
%                         Default is @sextractor.
%                         The program input/output: Cat=Prog(Image,Pars).
%                         Where Cat is a structure array containing:
%                         .Cat, .Col, .ColCell
%            'ProgPars' - A cell array of additional parameters to pass to
%                         ProgName. Default is {}.
%            'ReadFTPars'- A cell array of additional parameters to pass to
%                         read_fitstable.m. Default is {}.
%            'OutSIM'   - Force output to be of a SIM class (true) or a
%                         structure array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - A structure image array (SIM) with the catalog field
%            populated.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=addcat2sim('*.fits');
%          Sim=addcat2sim('*.fits','ReadImage',false'); % don't read images
% Reliable: 2
%--------------------------------------------------------------------------



%ImageField     = 'Im';
%HeaderField    = 'Header';
FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';

DefV.RePop            = true;
DefV.UseSimImageName  = false;
DefV.ColSort          = 'YWIN_IMAGE'; % either empty, scalar or string
DefV.ImageName        = {};
DefV.FITStable        = {};
DefV.ProgName         = @mextractor;
DefV.ProgPars         = {};
DefV.ReadFTPars       = {};
DefV.OutSIM           = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Sim  = images2sim(Sim,varargin{:});
Nsim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

if (~isempty(InPar.ImageName)),
    if (~iscell(InPar.ImageName)),
        [~,InPar.ImageName] = create_list(InPar.ImageName,NaN);
    end
end
if (~isempty(InPar.FITStable)),
    if (~iscell(InPar.FITStable)),
        [~,InPar.FITStable] = create_list(InPar.FITStable,NaN);
    end
end


for Isim=1:1:Nsim,
    if (~isfield(Sim,CatField)),
        InPar.RePop = true;
    else
        if (isempty(Sim(Isim).(CatField)) && isempty(Sim(Isim).(CatColField))),
            % Cat field is not populated
            InPar.RePop = true;
        end
    end
    
    if (InPar.RePop),        
            
        if (~isempty(InPar.ImageName)),
            Cat = InPar.ProgName(InPar.ImageName{Isim},InPar.ProgPars{:});
        else
            if (~isempty(InPar.FITStable)),
                Cat = read_fitstable(InPar.FITStable{Isim},InPar.ReadFTPars{:});
            else
                if (isfield(Sim(Isim),FileField) && InPar.UseSimImageName),
                    if (~isempty(Sim(Isim).(FileField))),
                        Cat = InPar.ProgName(Sim(Isim).(FileField),InPar.ProgPars{:});
                    else
                        % run ProgName directly on image matrix
                        % error('Can not read catalog into SIM');
                        Cat = InPar.ProgName(Sim(Isim),InPar.ProgPars{:});
                    end
                else
                    % run ProgName directly on image matrix
                    % error('Can not read catalog into SIM');
                    Cat = InPar.ProgName(Sim(Isim),InPar.ProgPars{:});
                end
            end
        end
        % populate SIM
        Sim(Isim).(CatField)        = Cat.(CatField);
        Sim(Isim).(CatColField)     = Cat.(CatColField);
        Sim(Isim).(CatColCellField) = Cat.(CatColCellField);
        
        if (~isempty(InPar.ColSort)),
            if (ischar(InPar.ColSort)),
                ColSort = Cat.(CatColField).(InPar.ColSort);
            else
                ColSort = InPar.ColSort;
            end
            Sim(Isim).(CatField) = sortrows(Sim(Isim).(CatField),ColSort);
        end
    end
end    