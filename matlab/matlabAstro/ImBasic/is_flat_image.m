function [IsFlat,IsNotSaturated]=is_flat_image(Sim,varargin)
%--------------------------------------------------------------------------
% is_flat_image function                                           ImBasic
% Description: Given a list of FITS images or SIM, look for good flat
%              field images. The search is done by looking for specific
%              keyword values in the image headers.
% Input  : - List of images to check for flat field images.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) A file contains a list of image (e.g., '@list').
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
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
%            'FieldName' - Field name containing the header in the
%                        structure returned by fitsinfo.m.
%                        Default is 'PrimaryData'. If empty then use
%                        default. If NaN then will attempt to look for 
%                        the correct field.
%            'Ind'     - Index of image header, in header structure.
%                        Default is 1.
%            'FlatKeyList' - List of header keywords which may contain
%                        the image type.
%                        Default is {'object','imgtype''imtype','type','imagetyp'}.
%                        All these keywords will be searched.
%            'FlatValList' - The list of expected values of the image type
%                        of a bias image. Default is
%                        {'flat','flatfield','domeflat','dome-flat','skyflat','sky-flat'}.
%            'CheckImage' - Check if the image is saturated {true|false}.
%                        Default is true.
%            'SatLevel'- Image saturation level. This is either a number
%                        or an header keyword name that containing
%                        the saturation level.
%                        Default is 60000.
%            'Verbose' - Print progress messages {true|false}.
%                        Default is false.
% Output : - A flag vector (IsFlat) indicating if an image have a flat
%            header keyword.
%          - A flag vector (IsNotSaturated) indicating if an image is
%            not saturated.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [IsFlat,IsNotSaturated]=is_flat_image('red*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

% input parameters
% images2sim.m parameters
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
% read header
DefV.FieldName  = [];
DefV.Ind        = 1;
% bias identification
DefV.FlatKeyList   = {'object','imgtype''imtype','type','imagetyp'};
DefV.FlatValList   = {'flat','flatfield','domeflat','dome-flat','skyflat','sky-flat'};
% check images
DefV.CheckImage    =  true;
DefV.SatLevel      = 60000;
DefV.Verbose       = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isstruct(Sim)),
    % do nothing
    InputSim = true;
    Nim      = numel(Sim);
else
    [~,ImageListCell] = create_list(Sim,NaN);
    InputSim = false;
    Nim      = numel(ImageListCell);
end


% go over images
IsFlat         = false(Nim,1);
IsNotSaturated = true(Nim,1);
for Iim=1:1:Nim,
    % read image
    if (InputSim);
        Header = Sim(Iim).(HeaderField);
        if (InPar.CheckImage),
            Image = Sim(Iim).(ImageField);
        end
    else
        % read from FITS
        Header = fits_header_cell(ImageListCell{Iim},InPar.FieldName,InPar.Ind);
        if (InPar.CheckImage),
            Image = fitsread(ImageListCell{Iim},InPar.FitsPars{:});
        end
    end
    
    % check candidate image header
    NewCellHead = cell_fitshead_getkey(Header,InPar.FlatKeyList,NaN);
    Vals  = NewCellHead(:,2);
    Nvals = numel(Vals);
    for Ivals=1:1:Nvals,
        if (~isempty(find(strcmpi(spacedel(Vals{Ivals}),InPar.FlatValList), 1))),
            IsFlat(Iim) = true;
        end
    end
    
    if (InPar.CheckImage),
        % check image saturation
        if (ischar(InPar.SatLevel)),
           NewCellHead = cell_fitshead_getkey(Header,InPar.SatLevel,NaN);
            if (~isnan(NewCellHead{1,2})),
                InPar.SatLevel = NewCellHead{1,2};
            else
                error('Saturation level is not available in header');
            end
        end
        IsNotSaturated(Iim) = all(Image(:)<InPar.SatLevel);
    end
end


if (InPar.Verbose),
    fprintf('Flat images search include total of %d images\n',Nim);
    if (InPar.CheckImage),
        fprintf('Found %d good flat images out of %d candidate flat images\n',length(find(IsFlat & IsNotSaturated)),length(find(IsFlat)));
    else
        fprintf('Found %d candidate flat images\n',length(find(IsFlat)));
    end
end
 

