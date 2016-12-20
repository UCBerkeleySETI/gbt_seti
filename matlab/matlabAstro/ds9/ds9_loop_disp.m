function ds9_loop_disp(List,varargin)
%-----------------------------------------------------------------------
% ds9_loop_disp function                                            ds9
% Description: Load a list of images (FITS file) into ds9 one by one.
%              Prompt the user for the next image.
%              Open ds9 before execution of this function.
% Input  : - List of images, see create_list.m for details.
%            Default is '*.fits'. If empty, use '*.fits'.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            Keywords and values can be one of the followings:
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
% Output : Null
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_disp('20050422051427p.fits',4,'InvertCM','yes')
% Reliable: 1
%-----------------------------------------------------------------------

Def.List = '*.fits';
if (nargin==0),
   List = Def.List;
end
if (isempty(List)),
    List = Def.List;
end
    

[~,ListCell] = create_list(List,NaN);
Nim = length(ListCell);
Flag = ones(Nim,1);
for Iim=1:1:Nim, 
   fprintf('Display image %s\n',ListCell{Iim});
   ds9_disp(ListCell{Iim},'first',varargin{:});
   fprintf('  Type Enter for next image; d to delete current image; r to remove from output list\n'); 
   R = input('Type return for next image: ','s');
   
   switch lower(R)
       case 'd'
           delete(ListCell{Iim});
       case 'r'
           Flag(Iim) = 0;
       otherwise
           % do nothing
   end
   
end

fprintf('\n');
FileName = input('Type file name in which to keep the non removed images, type enter to skip" ','s');
if (length(FileName)==0),
    % skip
else
    FID = fopen(FileName,'w');
    for Iim=1:1:Nim,
        if (Flag(Iim))
           fprintf(FID,'%s\n',ListCell{Iim});
        end
    end
end
