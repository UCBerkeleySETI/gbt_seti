function ds9_print(FileName,FileType,Palette,Level,Resolution)
%------------------------------------------------------------------------------
% ds9_print function                                                       ds9
% Description: Print the current frame to PostScript or JPG file.
% Input  : - File name to create.
%          - File type {'ps'|'jpg'|'tiff'|'png'|'ppm'|'fits'},
%            default is 'ps'.
%          - Color palette {'rgb'|'cmyk'|'gray'}, default is 'gray'.
%          - PostScript level {1|2}, default is '2'
%          - PostScript resolution {53|72|75|150|300|600}, default is 150.
% Output : null
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_frame(1); ds9_print;
% Reliable: 2
%------------------------------------------------------------------------------
DefFileType   = 'ps';
DefPalette    = 'gray';
DefLevel      = 2;
DefResolution = 150;

if  (nargin==1),
   FileType   = DefFileType;
   Palette    = DefPalette;
   Level      = DefLevel;
   Resolution = DefResolution;
elseif  (nargin==2),
   Palette    = DefPalette;
   Level      = DefLevel;
   Resolution = DefResolution;
elseif  (nargin==3),
   Level      = DefLevel;
   Resolution = DefResolution;
elseif  (nargin==4),
   Resolution = DefResolution;
elseif  (nargin==5),
   % do nothing
else
   error('Illegal number of input arguments');
end

switch lower(FileType),
 case 'ps'
    ds9_system(sprintf('xpaset -p ds9 print resolution %d',Resolution));
    ds9_system(sprintf('xpaset -p ds9 print level %d',Level));
    ds9_system(sprintf('xpaset -p ds9 print palette %s',lower(Palette)));
    ds9_system(sprintf('xpaset -p ds9 print destination file'));
    ds9_system(sprintf('xpaset -p ds9 print filename %s',FileName));
    ds9_system(sprintf('xpaset -p ds9 print'));
 case {'jpg','jpeg'}
    ds9_system(sprintf('xpaset -p ds9 saveimage jpeg %s',FileName));
 case {'tiff','png','ppm'}
    ds9_system(sprintf('xpaset -p ds9 saveimage %s %s',lower(FileType),FileName));
 case {'fits'}
    ds9_system(sprintf('xpaset -p ds9 savefits %s',FileName));
 otherwise
    error('Unknown FileType option');
end
