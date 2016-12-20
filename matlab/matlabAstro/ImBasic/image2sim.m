function [Sim,ImageNameFITS]=image2sim(Image,varargin)
%--------------------------------------------------------------------------
% image2sim function                                               ImBasic
% Description: Read a single image to structure image data (SIM).
%              For multiple files version of this program see
%              images2sim.m.
% Input  : - A single image in one of the following forms:
%            (1) A structure array (SIM).
%            (2) A file name.
%            (3) A matrix.
%            (4) A cell array with a single file name string.
%            (5) A cell array with a single matrix image.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'SIM'    - A SIM structure array to which to add the images.
%                       The first image will be added to SIM(1) and so on.
%                       This is useful when you want to read several types
%                       of images to the same SIM element
%                       (e.g., image, error image, flag image).
%                       Default is empty.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            'R2Field'- Name of the structure array field in which to
%                       store the image. Default is 'Im'.
%            'R2Name' - Name of the structure array field in which to
%                       store the image name. Default is 'ImageFileName'.
%            'ReadImage' - Read image {true|false}.
%                       Default is true.
%                       This option may be used only in case of a FITS
%                       image input. If false, then only the FITS header
%                       will be read.
%            'ReadHead'- Read header information {true|false}.
%                       Default is true.
%            'ReadHeadMethod' - Read header using fitsinfo.m (false) or
%                       fits_get_head.m (true). Default is true.
%            'HeadHDU' - Header HDU number. Default is 1.
%            'OnlyHead'- The name of the field name that contains
%                       the header keyword in the structure returned by
%                       fitsinfo.m. Default is 'PrimaryData'.
%                       If empty, then will save the entire fitsinfo
%                       structure. Relevant only for ReadHeadMethod=false.
%            'ImType' - Image type. One of the following:
%                       'FITS'   - fits image (default).
%                       'imread' - Will use imread to read a file.
%                       'mat'    - A matlab file containing matrix,
%                                  or structure array.
%                       'hdf5a'  - HDF5 file in which the image is stored
%                                  in a 'cube' field.
%            'FitsPars'- Cell array of additonal parameters to pass to
%                        fitsread.m. Default is {}.
%            'ImreadPars' - Cell array of additonal parameters to pass to
%                        imread.m. Default is {}.
%            'ImSec'   - Image section to read [xmin, xmax, ymim ymax].
%                        If empty, read entire image. Default is empty.
% Output : - SIM structure.
%          - FITS image name. Empty if image is not a FITS image.
% See also: images2sim.m, sim2file.m
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=image2sim('lred0121.fits');
%          Sim=image2sim('lred0122.fits','R2Field','Mask','SIM',Sim,'R2Name','MaskName');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField     = 'Im';
HeaderField    = 'Header';
FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
%CatField        = 'Cat';
%CatColField     = 'Col';
%CatColCellField = 'ColCell';

DefV.SIM        = [];
DefV.OutSIM     = true;
DefV.R2Field    = ImageField;
DefV.R2Name     = FileField;
DefV.ReadImage  = true;
DefV.ReadHead   = true;
DefV.OnlyHead   = 'PrimaryData';
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
DefV.ImSec      = [];
DefV.ReadHeadMethod = true;
DefV.HeadHDU    = 1;
%DefV.PopCat     = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


ImageNameFITS = [];
if (isstruct(Image) || issim(Image)),
    % no need to read image
    Sim = Image;
else
    
    ImageField = InPar.R2Field;  % set ImageField according to user specifications
    FileField  = InPar.R2Name;   % set FileField according to user specifications

    if (~isempty(InPar.SIM)),
        Sim   = InPar.SIM;
        InPar = rmfield(InPar,'SIM');
    end



    % deal with Image input
    if (iscell(Image)),
        % convert cell to a non-cell
        % either string or matrix are allowed
        Image = Image{1};
    end


    Nim = 1;
    if (InPar.OutSIM),
        Sim = simdef(Nim); %SIM;    % output is of SIM class
    else
        Sim = struct(InPar.ImageField,cell(Nim,1));  % output is structure array
    end


    if (ischar(Image)),
        switch lower(InPar.ImType)
            case 'fits'
                % read image from FITS file
                ImageNameFITS = Image;   % save FITS image name
                if (InPar.ReadImage),
                    if (isempty(InPar.ImSec)),
                        % read entire image
                        Sim.(ImageField)  = fitsread(Image,InPar.FitsPars{:});
                    else
                        % read only section of an image
                        %Sim.(ImageField)  = fitsread_section(Image,InPar.ImSec([1,3]),InPar.ImSec([2,4]));
                        Sim.(ImageField)  = fitsread_section(Image,InPar.ImSec([3,1]),InPar.ImSec([4,2]));
                    end
                end
                if (InPar.ReadHead),
                    if (InPar.ReadHeadMethod),
                        Sim.(HeaderField) = fits_get_head(Image,InPar.HeadHDU);
                    else
                        Head = fitsinfo(Image);
                        if (isempty(InPar.OnlyHead)),
                            % save the entire header from fitsinfo
                            Sim.(HeaderField) = Head; 
                        else
                            % save only the header
                            Sim.(HeaderField) = Head.(InPar.OnlyHead).Keywords;
                        end
                    end
                end
            case 'hdf5a'
                Tmp = loadh(Image);
                if (isfield(Tmp,'cube')),
                    Sim.(ImageField) = Tmp.cube;
                else
                    Sim.(ImageField) = [];
                end
                
            case 'imread'
                Sim.(ImageField)  = imread(Image,InPar.ImreadPars{:});
                if (~isempty(InPar.ImSec)),
                    Sim.(ImageField) = Sim.(ImageField)(InPar.ImSec(3):InPar.ImSec(4),InPar.ImSec(1):InPar.ImSec(2));
                end
            case 'mat'
                Sim.(ImageField)  = load2(Image);
                if (~isempty(InPar.ImSec)),
                    Sim.(ImageField) = Sim.(ImageField)(InPar.ImSec(3):InPar.ImSec(4),InPar.ImSec(1):InPar.ImSec(2));
                end
            otherwise
                error('Unknown ImType option');
        end
        Sim.(FileField) = Image;

    elseif (isnumeric(Image)),
        % Image in matrix form
        Sim.(ImageField) = Image;
        Sim.(FileField)  = '';

    elseif (isstruct(Image) || isa(Image,'SIM')),
        % Image already in structure array format
        Sim = Image;

    else
        error('Unknown Image format');
    end
    clear Image;

    % build source catalog of image
%     if (InPar.PopCat),
%         error('PopCat option not supported yet');
%     end
    
end