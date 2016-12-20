function [ListCell,MatCell,IsNumeric,Nim] = imget_fits(List,varargin)
%------------------------------------------------------------------------------
% imget_fits function                                                  ImBasic
% Description: Load a list of FITS images or mat files into a cell array
%              of images. This is mainly used as a utility program by
%              some of the *_fits.m function.
% Input  : - Either a list of images (see create_list.m for details)
%            or a cell array of matrices or a matrix.
%          * Arbitrary number of pairs of arguments ...,keyword,value,...
%            Possible keywords are:
%            'ReadAll'   - Read images to a matlab cell array {'y' | 'n'},
%                          default is 'n'.
%            'Section'   - Option to read only a section of the image.
%                          If empty matrix (default), then will read entire
%                          image.
%                          The format of this variable may be either
%                          section boundry or section center.
%            'SectionType' - Format for 'Section' which is one of the
%                            followings:
%                           'boundry' - [xmin xmax ymin ymax], default.
%                           'center'  - [xcenter ycenter xhalfsize yhalfsize]
%            'FitsReadPar' - A cell array of additional parameters to pass
%                            to fitsread.m (e.g., extension).
%                            Default is {}.
%            'FileType'    - File type to read {'FITS' | 'mat'}. Default is
%                            'FITS'.
%            'Cell'        - {'y' | 'n'}. If 'y' (default), then the second
%                            output argumemt 'MatCell' will be a cell of
%                            matrices. If 'n' and if 'MatCell' contains
%                            only one element than will return it as a
%                            matrix.
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

DefV.ReadAll = 'n';
DefV.Section = [];
DefV.SectionType = 'boundry';
DefV.FitsReadPar = {};
DefV.FileType    = 'FITS';  % {'FITS' | 'mat'}
DefV.Cell        = 'y';
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

if (isnumeric(List)),
    List = {List};
end

IsNumeric = 0;
MatCell = cell(1,0);
ListCell = cell(1,0);
if (iscell(List)),
    if (isnumeric(List{1})),
        % Image in matrix format
        MatCell = List;
        clear List;
        IsNumeric = 1;
    end
end

if (IsNumeric==0),
    [~,ListCell] = create_list(List,NaN);
    switch lower(InPar.ReadAll)
        case 'y'
            Nim = length(ListCell);
            MatCell = cell(Nim,1);
            for Iim=1:1:Nim,
                switch lower(InPar.FileType)
                    case 'fits'
                        MatCell{Iim} = fitsread(ListCell{Iim},InPar.FitsReadPar{:});
                    case 'mat'
                        Tmp = load(ListCell{Iim});
                        FN = fieldnames(Tmp);
                        if (length(FN)>1),
                            error(sprintf('mat %s file contains more than one variable',ListCell{Iim}));
                        else 
                           MatCell{Iim} = Tmp.(FN{1});
                        end
                    otherwise
                        error('Unknown FileType option');
                end
                
                if (~isempty(InPar.Section)),
                   [SubImage]=cut_image(MatCell{Iim},InPar.Section,InPar.SectionType);
                   MatCell{Iim} = SubImage;
                end
            end
        otherwise
            % do nothing
    end
end

if (IsNumeric==1),
    Nim = length(MatCell);
else
    Nim = length(ListCell);
end


switch lower(InPar.Cell)
    case 'n'
       if (Nim==1),
          MatCell = MatCell{1};
       else
          error('Can not return matrix instead of cell because more than one images was requested');
       end
    otherwise
        % do nothing
end



