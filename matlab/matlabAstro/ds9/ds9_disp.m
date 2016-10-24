function ds9_disp(ImageName,Frame,varargin)
%--------------------------------------------------------------------------
% ds9_disp function                                                    ds9
% Description: Load images (FITS file, image or matlab matrix) into ds9.
%              The user should open ds9 before execution of this function.
% Input  : - Image name - one of the followings:
%            A string containing a FITS file name to load, or a string
%            containing wild cards (see create_list.m for options).
%            A string that start with "@" in this case the string will
%            be regarded as a file name containing a list of images.
%            A matrix containing an image.
%            A cell array of FITS file names.
%            A cell array of matrices.
%            If empty, then only change the properties of the chosen frame.
%          - Frame - This can be a function {'new'|'first'|'last'|'next'|'prev'}
%            or a number, default is empty. If empty use current frame.
%            If the image name is a cell array then
%            the frame number can be a vector, otherwise it will
%            load the images to the sucessive frames.
%            Default is next frame.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            Keywords and values can be one of the followings:
%            'StartDS9'    - Start ds9. Default is true.
%            'Scale'       - Scale function {'linear'|'log'|'pow'|'sqrt'
%                                            'squared'|'histequ'}
%                            default is 'linear'.
%            'ScaleLimits' - Scale limits [Low High], no default.
%            'ScaleMode'   - Scale mode {percentile |'minmax'|'zscale'|'zmax'},
%                            default is 'zscale'.
%            'CMap'        - Color map {'Grey'|'Red'|'Green'|'Blue'|'A'|'B'|
%                                       'BB'|'HE'|'I8'|'AIPS0'|'SLS'|'HSV'|
%                                       'Heat'|'Cool'|'Rainbow'|'Standard'|
%                                       'Staircase'|Color'},
%                            default is 'Grey'.
%             'InvertCM'   - Invert colormap {'yes'|'no'}, default is 'no'.
%             'Rotate'     - Image rotation [deg], default is to 0 deg.
%             'Orient'     - Image orientation (flip) {'none'|'x'|'y'|'xy'},
%                            default is 'none'.
%                            Note that Orient is applayed after rotation.
%             'Zoom'       - Zoom to 'fit' or [XY] or [X Y] zoom values,
%                            default is [2] (i.e., [2 2]).
%             'Pan'        - Center image on coordinates [X,Y],
%                            default is [] (do nothing).
%             'PanCoo'     - Coordinate system for Pan
%                            {'fk4'|'fk5'|'icrs'|'iamge'|'physical'},
%                            default is 'image'.
%             'Match'      - If more than one image is loaded, then
%                            match (by coordinates) all the images to
%                            the image in the first frame.
%                            Options are
%                            {'wcs'|'physical'|'image'|'none'},
%                            default is 'none'.
%             'MatchCB'    - If more than one image is loaded, then
%                            match (by colorbars) all the images to
%                            the image in the first frame.
%                            Options are {'none'|'colorbars'}
%                            default is 'none'.
%             'MatchS'     - If more than one image is loaded, then
%                            match (by scales) all the images to
%                            the image in the first frame.
%                            Options are {'none'|'scales'}
%                            default is 'none'.
%             'Tile'       - Tile display mode {'none'|'column'|'row'} or
%                            the number of [row col], default is 'none'.
% Output : null
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_disp('20050422051427p.fits',4,'InvertCM','yes')
% Reliable: 2
%--------------------------------------------------------------------------

ImageField    = 'Im';
HeaderField   = 'Header';

DefFrame = []; %'new';
if (nargin==1),
   Frame = DefFrame;
end
if (isempty(Frame)==1),
   Frame = DefFrame;
end

CurDir = pwd;

%--------------------------
%--- Set default values ---
%--------------------------
DefV.StartDS9     = true;
DefV.Scale        = 'linear';
DefV.ScaleLimits  = [];
DefV.ScaleMode    = 'zscale';
DefV.CMap         = 'Grey';
DefV.InvertCM     = 'no';
DefV.Rotate       = 0;
DefV.Orient       = 'none';
DefV.Zoom         = 2;
DefV.Pan          = [];
DefV.PanCoo       = 'image';
DefV.Match        = 'none';
DefV.MatchCB      = 'none';
DefV.MatchS       = 'none';
DefV.Tile         = 'none';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (InPar.StartDS9),
    ds9_start;
end

%-----------------------------
%--- convert to cell array ---
%-----------------------------
DelTempFile = false;

if (isempty(ImageName)),
   % change properties of frame
   Nim = 1;
   Images = [];
else
    if (ischar(ImageName)),
       % load new image into frame
       [~,Images] = create_list(ImageName,NaN);
       
       Nim = length(Images);   % number of frames
    elseif (iscell(ImageName)),
        if (iscellstr(ImageName)),
            % cell array of strings (images)
            Images = ImageName;
            Nim    = numel(Images);
        else
            % cell array of matrices
            Nim = numel(ImageName);
            Images = cell(Nim,1);
            for Iim=1:1:Nim,
                Images{Iim} = sprintf('%s.fits',tempname);
                fitswrite_my(ImageName{Iim},Images{Iim});
                CurDir = '';
            end
            DelTempFile = true;
        end
    elseif (isnumeric(ImageName)),
        % assume a matrix
        Images{1} = sprintf('%s.fits',tempname);
        TempFile  = Images{1};
        fitswrite_my(ImageName,TempFile);
        Nim         = 1;
        DelTempFile = true;
        CurDir      = '';
    elseif (isstruct(ImageName) || issim(ImageName)),
        Nim = numel(ImageName);
        Images = cell(Nim,1);
        for Iim=1:1:Nim,
            Images{Iim} = sprintf('%s.fits',tempname);
            if (isfield_notempty(ImageName(Iim),HeaderField)),
                Head = ImageName(Iim).(HeaderField);
            else
                Head = [];
            end
            fitswrite_my(ImageName(Iim).(ImageField),Images{Iim},Head);
            CurDir = '';
        end
        DelTempFile = true;
    else
        error('Unknown image input type');
    end
end


if (ischar(Frame)),
   if (numel(Frame)==1),
      Frame = (1:1:length(Frame)).'; %Frame.*ones(Nim,1);
   end
end
switch InPar.PanCoo
 case {'fk4','fk5','icrs'}
    InPar.PanCoo = sprintf('wcs %s',InPar.PanCoo);
 otherwise
    % do nothing
end

if (Nim>1 && isempty(Frame)),
    Frame = (1:1:Nim).';
end


for Iim=1:1:Nim,
   %--------------------
   %--- create frame ---
   %--------------------
   if (isempty(Frame)),
       % do noting
   else
       if (ischar(Frame)),
           if (Iim==1),
              %CF=ds9_get_filename;
              %strcmp(CF,'  ')
              %if strcmp(CF,'  ')
              %   ds9_system(sprintf('xpaset -p ds9 frame 1'));
              %else
              ds9_system(sprintf('xpaset -p ds9 frame %s',Frame));
              %end
           else
              ds9_system(sprintf('xpaset -p ds9 frame %s',Frame));
           end
       else
          ds9_system(sprintf('xpaset -p ds9 frame frameno %d',Frame(Iim)));
       end
   end
   %------------------
   %--- Load image ---
   %------------------
   if (isempty(Images)),
      % use existing image
   else
      % load image
      if (ischar(Images{Iim})),
         % load FITS file name
        
         ds9_system(sprintf('xpaset -p ds9 file %s%s%s',CurDir,filesep,Images{Iim}));
      else
         % load matrix
      end
   end

   %-----------------------
   %--- set image scale ---
   %-----------------------
   ds9_system(sprintf('xpaset -p ds9 scale %s',InPar.Scale));
   if (ischar(InPar.ScaleMode)),
      ds9_system(sprintf('xpaset -p ds9 scale mode %s',InPar.ScaleMode));
   else
      ds9_system(sprintf('xpaset -p ds9 scale mode %f',InPar.ScaleMode));
   end
   if (isempty(InPar.ScaleLimits)),
      % do nothing
   else
      ds9_system(sprintf('xpaset -p ds9 scale limits %f %f',InPar.ScaleLimits));
   end

   %---------------------
   %--- Set color map ---
   %---------------------
   ds9_system(sprintf('xpaset -p ds9 cmap %s',InPar.CMap));
   ds9_system(sprintf('xpaset -p ds9 cmap invert %s',InPar.InvertCM));

   %--------------------------------------
   %--- Image orientation and rotation ---
   %--------------------------------------
   ds9_system(sprintf('xpaset -p ds9 rotate to %f',InPar.Rotate));
   ds9_system(sprintf('xpaset -p ds9 orient %s',InPar.Orient));

   %------------------
   %--- Image zoom ---
   %------------------
   if (ischar(InPar.Zoom)),
      ds9_system(sprintf('xpaset -p ds9 zoom to %s',InPar.Zoom));
   else
      if (numel(InPar.Zoom)==1),
         InPar.Zoom = [InPar.Zoom InPar.Zoom];
      end
      ds9_system(sprintf('xpaset -p ds9 zoom to %f %f',InPar.Zoom));
   end

   %-----------------
   %--- Image pan ---
   %-----------------
   if (isempty(InPar.Pan)),
      % do nothing
   else
      ds9_system(sprintf('xpaset -p ds9 pan to %f %f %s',InPar.Pan(1),InPar.Pan(2),InPar.PanCoo));
   end
end


%--------------------
%--- Match frames ---
%--------------------
switch lower(InPar.Match),
 case 'none'
    % do nothing
 otherwise
    ds9_system(sprintf('xpaset -p ds9 frame first'));
    ds9_system(sprintf('xpaset -p ds9 match frames %s',InPar.Match));

    % return focus to last frame
    ds9_system(sprintf('xpaset -p ds9 frame last'));
end

switch lower(InPar.MatchCB),
 case 'none'
    % do nothing
 otherwise
    ds9_system(sprintf('xpaset -p ds9 frame first'));
    ds9_system(sprintf('xpaset -p ds9 match %s',InPar.MatchCB));

    % return focus to last frame
    ds9_system(sprintf('xpaset -p ds9 frame last'));
end

switch lower(InPar.MatchS),
 case 'none'
    % do nothing
 otherwise
    ds9_system(sprintf('xpaset -p ds9 frame first'));
    ds9_system(sprintf('xpaset -p ds9 match %s',InPar.MatchS));

    % return focus to last frame
    ds9_system(sprintf('xpaset -p ds9 frame last'));
end


%---------------------
%--- Set Tile mode ---
%---------------------
switch lower(InPar.Tile)
 case 'none'
    % do nothing
 otherwise
    if (ischar(InPar.Tile)),
       ds9_system(sprintf('xpaset -p ds9 tile mode %s',InPar.Tile));
    else
       ds9_system(sprintf('xpaset -p ds9 tile grid layout %d %d',InPar.Tile));
    end
end

if (DelTempFile)
    delete_cell(Images);
end

