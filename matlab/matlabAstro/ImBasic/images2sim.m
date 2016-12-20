function [Sim,ImageNameFITS]=images2sim(Images,varargin)
%--------------------------------------------------------------------------
% images2sim function                                              ImBasic
% Description: Read multiple images to structure array image data (SIM).
%              For single file version of this program see
%              image2sim.m.
% Input  : - Multiple images in one of the following forms:
%            (1) A structure array (SIM).
%            (2) A string containing wild cards (see create_list.m).
%            (3) A cell array of matrices.
%            (4) A file name that contains a list of files
%                (e.g., '@list', see create_list.m)
%            (5) A cell array of file names.
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
%            'ImType' - Image type. One of the following:
%                       'FITS'   - fits image (default).
%                       'imread' - Will use imread to read a file.
%                       'mat'    - A matlab file containing matrix,
%                                  or structure array.
%            'FitsPars'- Cell array of additonal parameters to pass to
%                        fitsread.m. Default is {}.
%            'ImreadPars' - Cell array of additonal parameters to pass to
%                        imread.m. Default is {}.
%            'ImSec'   - Image section to read [xmin, xmax, ymim ymax].
%                        If empty, read entire image. Default is empty.
% Output : - SIM structure.
%          - Cell array of FITS image names. Element is empty if image
%            is not a FITS image.
% See also: image2sim.m, sim2file.m
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=images2sim('lred012*.fits');
%          Sim=images2sim('lred012*.fits','R2Field','Mask','SIM',Sim,'R2Name','MaskName');
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.SIM        = [];
DefV.OutSIM     = true;
DefV.R2Field    = ImageField;
DefV.R2Name     = FileField;
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
DefV.ImSec      = [];

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});



if (isstruct(Images) || issim(Images)),
    % no need to read images
    Sim = Images;
    if (nargout>1),
        Nim = numel(Sim);
        ImageNameFITS = cell(1,Nim);
    end
else

    ImageField = InPar.R2Field;  % set ImageField according to user specifications
    FileField  = InPar.R2Name;   % set FileField according to user specifications

    if (isnumeric(Images)),
        Images = {Images};
    end

    if (~isempty(InPar.SIM)),
        Sim = InPar.SIM;
        %InPar.SIM = [];
        % add new fields to Sim if needed
        if (~isfield(Sim,InPar.R2Field)),
            [Sim.(InPar.R2Field)] = deal(cell(size(Sim)));
        end
        if (~isfield(Sim,InPar.R2Name)),
            [Sim.(InPar.R2Name)] = deal(cell(size(Sim)));
        end
    end

    if (ischar(Images)),
        [~,Images] = create_list(Images,NaN);
    end


    Nim = length(Images);
    ImageNameFITS = cell(1,Nim);
    for Iim=1:1:Nim,
        % Images now contains either a cell array or a structure array.
        if (isempty(InPar.SIM)),
            SIM_e = [];
        else
            SIM_e = InPar.SIM(Iim);
        end
        [Sim(Iim),ImageNameFITS{Iim}] = image2sim(Images(Iim),varargin{:},'SIM',SIM_e);
    end

end
