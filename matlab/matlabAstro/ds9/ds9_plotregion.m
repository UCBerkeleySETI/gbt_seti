function FileName=ds9_plotregion(X,Y,varargin)
%------------------------------------------------------------------------------
% ds9_plotregion function                                                  ds9
% Description: Write a region file containing various plots and load it
%              to the ds9 display.
% Input  : - Vector of X coordinate (pixels or degrees).
%          - Vector of Y coordinate (pixels or degress).
%          * Arbitrary number of pairs of input arguments:...,keyword,value,...
%            Following keywords are available:
%            'FileName'   - Region file name, default is to create a temp file.
%            'Save'       - save region file {'y' | 'n'}, default is 'n'.
%            'Load'       - Load region file into ds9 display {'y'|'n'},
%                           default is 'y'.
%            'Append'     - Append to an existing region file {'y'|'n'},
%                           default is 'n'.
%                           If 'y' then will not write the region file header.
%            'Coo'        - Coordinate type {'image'|'fk5'},
%                           default is 'image'.
%            'Type'       - Symbol type {'circle'|'box'|'ellipse'|'vector'|
%   		                'line'|'polygon}, default is 'circle'.
%            'Color'      - Symbol color, default is 'red'.
%            'Width'      - Marker line width, default is 2.
%            'Size'       - The marker size descriptors.
%                           These can be a single line vector or a matrix.
%                           If matrix then use each line for each X/Y position.
%                           Following attributes are needed:
%                           For 'circle':  [Radius]
%                           For 'box':     [Width,Height]
%                           For 'ellipse': [Major axis, Minor axis]
%                           For 'vector':  [Length,PA]
%                           For 'line':    [StartX,StartY,EndX,EndY]
%                           For 'polygon': [X1,Y1,X2,Y2,X3,Y3...]
%                           The units of coordinates are either pixels
%                           or arcseconds for 'image' and 'fk5', respectively.
%                           The units of length are either pixels or arcsec.
%                           The units of PA are degrees.
%            'Text'       - String of text associated with symbol.
%                           Default is ''.
%            'Font'       - Text fonts. Default is 'helvetica 16 normal'.
% Output : - Region file name.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Feb 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example:
% ds9_plotregion(150.02,41.02,'coo','fk5','type','line','size',[150.02,41.02,150.04,41.04]);
% Reliable: 2
%------------------------------------------------------------------------------

DefV.FileName = tempname;
DefV.Save     = 'n';
DefV.Load     = 'y';
DefV.Append   = 'n';
DefV.Coo      = 'image';
DefV.Type     = 'circle';
DefV.Size     = 10;
DefV.Color    = 'red';
DefV.Width    = 2;
DefV.Text     = '';
DefV.Font     = 'helvetica 16 normal';


Par = set_varargin_keyval(DefV,'y','use',varargin{:});

% check if region file exist
if (exist(Par.FileName,'file')==0),
   switch lower(Par.Append)
    case 'y'
       error('user requested to append region file, but file doesnt exist');
    otherwise
       % do nothing
   end
end

switch lower(Par.Append)
 case 'n'
    % write header of region file
    FID = fopen(Par.FileName,'w');
    fprintf(FID,'# Region file format: DS9 version 4.1\n');
    fprintf(FID,'# Written by Eran Ofek via ds9_plotregion.m\n');
    fprintf(FID,'global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n');
    fprintf(FID,'%s\n',Par.Coo);
 case 'y'
    % append header to an existing region file
    FID = fopen(Par.FileName,'a');
 otherwise
    error('Unknown Append option');
end

switch lower(Par.Coo)
 case 'image'
    CooUnits = ''; 
 case 'fk5'
    CooUnits = '"';
 otherwise
    error('Coo units is not supported');
end

Nreg = length(X);
if (size(Par.Size,1)==1),
   Par.Size = repmat(Par.Size,Nreg,1);
end
switch lower(Par.Type)
 case 'line'
    Nreg = size(Par.Size,1);
end 


Nsize = size(Par.Size,2);
for Ireg=1:1:Nreg,

   switch lower(Par.Type)
    case {'circle'}
           fprintf(FID,'%s(%15.8f,%15.8f,%15.8f%s)',... 
  		   Par.Type,X(Ireg),Y(Ireg),Par.Size(Ireg),CooUnits);
    case {'box','ellipse'}
            fprintf(FID,'%s(%15.8f,%15.8f,%15.8f%s,%15.8f%s,%9.5f)',... 
	       Par.Type,X(Ireg),Y(Ireg),Par.Size(Ireg,1),CooUnits,Par.Size(Ireg,2),CooUnits,Par.Size(Ireg,3));
    case {'vector'}
           fprintf(FID,'# %s(%15.8f,%15.8f,%15.8f%s,%9.5f)',... 
 	       Par.Type,X(Ireg),Y(Ireg),Par.Size(Ireg,1),CooUnits,Par.Size(Ireg,2));
    case {'line'}
           fprintf(FID,'%s(%15.8f,%15.8f,%15.8f,%15.8f)',... 
 	       Par.Type,Par.Size(Ireg,1),Par.Size(Ireg,2),Par.Size(Ireg,3),Par.Size(Ireg,4));
    case {'polygon'}
          fprintf(FID,'%s(',Par.Type);
          for Isize=1:1:length(X),
              fprintf(FID,'%15.8f,%15.8f,',X(Isize),Y(Isize)); %Par.Size(Ireg,Isize));
          end
          fprintf(FID,'%15.8f,%15.8f)',X(Isize),Y(Isize)); %Par.Size(Ireg,Nsize));
    otherwise
       error('Unknown Type option');
   end
 
   fprintf(FID,'# color=%s width=%d font="%s" text={%s}\n',...
               Par.Color,Par.Width,Par.Font,Par.Text);

end

fclose(FID);

switch lower(Par.Load)
 case 'y'
    % load to ds9
    ds9_regions('load',Par.FileName);
 case 'n'
    % do nothing
 otherwise
    error('Unknown Load option');
end

switch lower(Par.Save)
 case 'y'
    % do nothing
 case 'n'
    delete(Par.FileName);
 otherwise
    error('Unknwon Save option');
end

FileName = Par.FileName;
