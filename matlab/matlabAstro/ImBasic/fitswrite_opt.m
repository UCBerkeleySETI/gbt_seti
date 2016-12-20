function OutImageName=fitswrite_opt(OutImageMatrix,OutImageName,HeaderInfo,varargin);
%--------------------------------------------------------------------------
% fitswrite_opt function                                           ImBasic
% Description: Write a FITS image using fitswrite_my.m. This a version
%              fitswrite_my.m with some additional options.
% Input  : - Image matrix.
%          - Image file name.
%          - Header cell array (see fitswrite_my.m for details).
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'InImage'  - Input image name (used if OutImageName is empty).
%                         Default is empty matrix.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'HeaderInfo'- (Original) image header. Default is empty
%                         matrix.
%            'CopyHead' - Copy header from (original) image {'y' | 'n'}.
%                         Default is 'y'.
%                         If 'n' then will not write HeaderInfo to the
%                         new header.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'AddHeadP' - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%                         This is used as an additional keywords from
%                         the calling function.
%            'DelDataSec'- Delete original DATASEC keyword from header
%                         {'y' | 'n'}, default is 'n'.
%            'DataType' - Output data type (see fitswrite_my.m for options), 
%                         default is 'float32'.
%            'Save'     - Save FITS image to disk {'y' | 'n' | 'mat'}.
%                         Default is 'y'.
% Output : - Output image file name.
% Tested : Matlab 7.13
%     By : Eran O. Ofek                    Apr 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%--------------------------------------------------------------------------


DefV.InImage    = [];
DefV.OutPrefix  = '';
DefV.OutDir     = '';
DefV.HeaderInfo = [];
DefV.CopyHead   = 'y';
DefV.AddHead    = [];
DefV.AddHeadP   = [];
DefV.DelDataSec = 'n';
DefV.DataType   = 'float32';
DefV.Save       = 'y';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


if (isempty(OutImageName)),
    % Ouput image name is not orovided
    % Use Inpat image name
    OutImageName = InPar.InImage;
end
if (isempty(OutImageName)),
    % InImage and OutImageName are not provided
    switch lower(InPar.Save)
        case 'y'
            error('When Save=y either input or output image name must be provided');
        otherwise
            % do nothing
    end
else
    OutImageName = sprintf('%s%s%s',InPar.OutDir,InPar.OutPrefix,OutImageName);
end
         

%-------------------
%--- write image ---
%-------------------
switch lower(InPar.Save)
    case {'y','fits'}
        %------------------------
        %--- write FITS image ---
        %------------------------

        %--- Add to header comments regarding file creation ---
        switch lower(InPar.CopyHead)
            case 'y'
                % do nothing
            case 'n'
                % delete HeaderInfo
                HeaderInfo = [];
            otherwise
                error('Unknown CopyHead option');
        end
                
        %[HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
        %                             InPar.AddHeadProg
        %                             Inf,'COMMENT','','Created by imtrans_1fits.m written by Eran Ofek');   
                                 
        if (~isempty(InPar.AddHead)),
           %--- Add additional header keywords ---
           HeaderInfo = [HeaderInfo; InPar.AddHead];
        end
        if (~isempty(InPar.AddHeadP)),
           %--- Add additional header keywords ---
           HeaderInfo = [HeaderInfo; InPar.AddHeadP];
        end

        switch lower(InPar.DelDataSec)
            case 'n'
                % do nothing
            case 'y'
                % delete DATASEC keyword from header
                [HeaderInfo] = cell_fitshead_delkey(HeaderInfo,'DATASEC');
            otherwise
                error('Unknown DelDataSec option');
        end

        %--- Write fits file ---
        fitswrite_my(OutImageMatrix,OutImageName); %,HeaderInfo,InPar.DataType);
        
    case 'mat'
        %----------------------
        %--- wtite mat file ---
        %----------------------
        save(OutImageName,'OutImageMatrix');
   
    case 'n'
           % do not write output
   
    otherwise
        error('Unknown Save option');
end
   
   
