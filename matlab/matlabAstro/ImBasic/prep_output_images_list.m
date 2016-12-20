function OutImagesCell=prep_output_images_list(InImages,OutImages,varargin)
%------------------------------------------------------------------------------
% prep_output_images_list function                                     ImBasic
% Description: Prepare the output images list by concanating the output
%              directory and prefix to file names.
%              This is a utility program mainly used by some of the
%              *_fits.m functions.
% Input  : - A list of images (see create_list.m for details)
%          * Arbitrary number of pairs of arguments ...,keyword,value,...
%            Possible keywords are:
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
% Output : - Cell array of images names.
%            Empty if input are matrices.
%          - Cell array in which each cell contains a loaded image matrix.
%            Empty if no need to read images.
%          - A flag indicating if the user input was a list of matrices (1)
%            or a list of files (0).
%          - Number of images in cell.
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Jun 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------


DefV.OutPrefix = '';
DefV.OutDir    = '';
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});   % CheckExist='n'

if (isempty(OutImages)),
    if (isnumeric(InImages)),
        error('Illegal InImages input type');
    else
       OutImages = InImages;
    end
end

[~,OutImagesCell] = create_list(OutImages,NaN);

Nim = length(OutImagesCell);

if (~isempty(InPar.OutPrefix) || ~isempty(InPar.OutDir)),
    if (isempty(InPar.OutDir)),
        FileSep = '';
    else
        FileSep = filesep;
    end
    for Iim=1:1:Nim,
        OutImagesCell{Iim} = sprintf('%s%s%s%s',InPar.OutDir,FileSep,InPar.OutPrefix,OutImagesCell{Iim});
    end
end
