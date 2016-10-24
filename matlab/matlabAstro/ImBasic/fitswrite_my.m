function [Flag,HeaderInfo]=fitswrite_my(Image,FileName,HeaderInfo,DataType,varargin)
%--------------------------------------------------------------------------
% fitswrite_my_old function                                        ImBasic
% Description: Write a simple 2D FITS image.
% Input  : - A 2D matrix to write as FITS file.
%          - String containing the image file name to write.
%          - Cell array containing header information to write to image.
%            The keywords: SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, EXTEND
%            will be re-written to header by default.
%            The keywords BSCALE and BZERO wil be written to header if
%            not specified in the header information cell array.
%            Alternatively this could be a character array (Nx80)
%            containing the header (no changes will be applied).
%            If not given, or if empty matrix (i.e., []) than write a
%            minimal header.
%          - DataType in which to write the image, supported options are:
%            'int8',8            
%            'int16',16
%            'int32',32
%            'int64',64
%            'single','float32',-32    (default)
%            'double','float64',-64            
%          * Arbitrary number of pairs of input arguments: 
%            ...,keyword,value,... - possible keys are:
%            'IdentifyInt'  - attempt to identify integers in header
%                             automaticaly and print them as integer.
%                             {'y' | 'n'}, default is 'y'.
%            'ResetDefKey'  - Reset default keys {'y' | 'n'},
%                             default is 'y'.
%                             Default keys are:
%                             {'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2',
%                              'EXTEND','BSCALE',BZERO'}.
%            'OverWrite'    - Over write existing image {'y' | 'n'},
%                             default is 'y'.
% Output : - Flag indicating if image was written to disk (1) or not (0).
%          - Actual header information written to file.
% See also: fitswrite_my.m
% Bugs   : Don't write standard FITS file
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Examples: [Flag,HeaderInfo]=fitswrite_my_old(rand(2048,1024),'Example.fits');
% Reliable: 2
%--------------------------------------------------------------------------


Def.HeaderInfo = [];
Def.DataType   = -32;
if (nargin==2),
   HeaderInfo = Def.HeaderInfo;
   DataType   = Def.DataType;
elseif (nargin==3),
   DataType   = Def.DataType;
end

% set default for additional keywords:
DefV.IdentifyInt = 'y';
DefV.ResetDefKey = 'y';
DefV.OverWrite   = 'y';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


Flag = 1;

switch DataType
 case {'int8',8}
    DataType = 8;
 case {'int16',16}
    DataType = 16;
 case {'int32',32}
    DataType = 32;
 case {'int64',64}
    DataType = 64;
 case {'single','float32',-32}
    DataType = -32;
 case {'double','float64',-64}
    DataType = -64;
 otherwise
    error('Unknown DataType option');
end



if (ischar(HeaderInfo)==0),
   
%--- Set the FITS "mandatory" keywords ---
ImSize = size(Image);
switch lower(InPar.ResetDefKey)
   case 'n'
      % do nothing
   case 'y'
      if (isempty(HeaderInfo)),
         % do nothing
      else
         % delete "default" keys:
         HeaderInfo = cell_fitshead_delkey(HeaderInfo,{'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND','BSCALE','BZERO'});
      end
      HeaderInfo = cell_fitshead_addkey(HeaderInfo,...
                                        0,'SIMPLE',true(1,1),'file does conform to FITS standard',...
                                        0,'BITPIX',int32(DataType),'number of bits per data pixel',...
                                        0,'NAXIS' ,int32(length(ImSize)),'number of data axes',...
                                        0,'NAXIS1',int32(ImSize(2)),'length of data axis 1',...
                                        0,'NAXIS2',int32(ImSize(1)),'length of data axis 2',...
                                        0,'EXTEND',false(1,1),'FITS dataset may contain extensions',...
                                        0,'BZERO' ,single(0),'offset data range to that of unsigned short',...
                                        0,'BSCALE',single(1),'default scaling factor');
   otherwise
      error('Unknown ResetDefKey option');
end

% check if last keyword is END - delete if needed.
if (strcmp(HeaderInfo{end,1},'END')),
    HeaderInfo = HeaderInfo(1:end-1,:);
end


%--- Write creation date to header ---
Time = get_atime([],0,0);
HeaderInfo = cell_fitshead_addkey(HeaderInfo,...
                                  'CRDATE',Time.ISO,'Creation date of FITS file',...
                                  'COMMENT','','File Created by MATLAB fitswrite.m written by Eran Ofek');

%--- Convert default keywords to int32 ---
KeysInt32 = {'BITPIX','NAXIS','NAXIS1','NAXIS2','BZERO','BSCALE'};
Ni32     = length(KeysInt32);
for Ii32=1:1:Ni32,
   I = find(strcmp(HeaderInfo(:,1),KeysInt32{Ii32})==1);
   if (~isempty(I)),
      HeaderInfo{I,2} = int32(HeaderInfo{I,2});
   end
end
     

%--- Convert default keywords to logical ---
KeysLogi = {'SIMPLE','EXTEND'};
Nlogi     = length(KeysLogi);
for Ilogi=1:1:Nlogi,
   I = find(strcmp(HeaderInfo(:,1),KeysLogi{Ilogi})==1);
   if (~isempty(I)),
      if (~islogical(HeaderInfo{I,2}))
         switch HeaderInfo{I,2}
          case 'F'
             HeaderInfo{I,2} = false(1,1); 
          case 'T'
             HeaderInfo{I,2} = true(1,1);
          otherwise
             error('Keyword type is not logical');
         end
      end
   end
end


% check if last keyword is END - if not add.
if (~strcmp(HeaderInfo{end,1},'END')),
    HeaderInfo{end+1,1} = 'END';
    HeaderInfo{end,2} = '';
    HeaderInfo{end,3} = '';
end

%--- Prepare string of header information to write to header ---
HeaderBlock = '';
[Nline,Nr] = size(HeaderInfo);

Counter = 0;
for Iline=1:1:Nline,
   if (~isempty(HeaderInfo{Iline,1}) && strcmpi(HeaderInfo{Iline,1},'END')==0),
      %--- write keyword name ---
      HeaderLine = sprintf('%-8s',upper(HeaderInfo{Iline,1}));
      switch upper(HeaderInfo{Iline,1})
       case {'COMMENT','HISTORY'}
          % do not write "=" sign
          HeaderLine = sprintf('%s',HeaderLine);
       otherwise
          % write "=" sign
          HeaderLine = sprintf('%s= ',HeaderLine);
      end
      %--- write keyword value ---
      switch upper(HeaderInfo{Iline,1})
       case {'COMMENT','HISTORY'}
          % do not write value
       otherwise
          if (ischar(HeaderInfo{Iline,2})),
             HeaderLine = sprintf('%s''%s''',HeaderLine,HeaderInfo{Iline,2});
             Nblanks    = 20-(length(HeaderInfo{Iline,2})+2);
   
             if (Nblanks<0),
                Nblanks = 0;
             end
             HeaderLine = sprintf('%s%s',HeaderLine,blanks(Nblanks));
         
          elseif (islogical(HeaderInfo{Iline,2}))
  	          switch HeaderInfo{Iline,2}
              case 1
	         HeaderLine = sprintf('%s%20s',HeaderLine,'T');
              case 0
	         HeaderLine = sprintf('%s%20s',HeaderLine,'F');                 
              otherwise
	         error('DataType is not logical');
             end
          elseif (isinteger(HeaderInfo{Iline,2}))
             HeaderLine = sprintf('%s%20i',HeaderLine,HeaderInfo{Iline,2});
          elseif (isfloat(HeaderInfo{Iline,2})),

             switch InPar.IdentifyInt
              case 'y'
                 % attempt to identify integers automatically
	             if (fix(HeaderInfo{Iline,2})==HeaderInfo{Iline,2}),
                    % integer identified - print as integer
                    HeaderLine = sprintf('%s%20i',HeaderLine,HeaderInfo{Iline,2});
                 else
                    % float or double
                    HeaderLine = sprintf('%s%20.8f',HeaderLine,HeaderInfo{Iline,2});
                 end
              case 'n'
                   % float or double
                   HeaderLine = sprintf('%s%20.8f',HeaderLine,HeaderInfo{Iline,2});
              otherwise
		             error('Unknown IdentifyInt option');
             end
          else
             error('unknown Format in header information');
          end
      end
      %--- write comment to header ---
      if (Nr>2),
         switch upper(HeaderInfo{Iline,1})
          case {'COMMENT','HISTORY'}
             % do not write "/"
             HeaderLine = sprintf('%s%-s',HeaderLine,HeaderInfo{Iline,3});
          otherwise
             if (isempty(HeaderInfo{Iline,3})),
                % do nothing - do not add comment
             else
                HeaderLine = sprintf('%s /%-s',HeaderLine,HeaderInfo{Iline,3});
             end
          end
       end
       % cut line if longer than 80 characters or paded with spaces
       if (length(HeaderLine)>80),
          HeaderLine = HeaderLine(1:80);
       end
       HeaderLine = sprintf('%-80s',HeaderLine);

       %%HeaderBlock = sprintf('%s%s',HeaderBlock,HeaderLine);
       Counter = Counter + 1;
       HeaderBlock(Counter,:) = sprintf('%s',HeaderLine);
    end   
end
%--- Add End keyword to end of header ---
%%HeaderBlock = sprintf('%s%-80s',HeaderBlock,'END');
Counter = Counter + 1;
HeaderBlock(Counter,:) = sprintf('%-80s','END');

else
  %--- HeaderInfo is already provided as char array ---
  % assume contains also the 'END' keyword
  HeaderBlock = HeaderInfo;
end


%--- pad by spaces up to the next 2880 character block ---
PadBlock = sprintf('%s',blanks(2880 - mod(numel(HeaderBlock),2880)));


%--- pad by spaces up to the next 2880 character block ---
%HeaderBlock = sprintf('%s%s',HeaderBlock,blanks(2880 - 80 -
%mod(length(HeaderBlock),2880)));
%--- Add End keyword to end of header ---
%HeaderBlock = sprintf('%s%-80s',HeaderBlock,'END');


%--- Open file for writing in rigth byte order in all platforms.
switch lower(InPar.OverWrite)
   case 'n'
      if (exist(FileName,'file')==0),
         % File does not exist - continue
      else
         error('Output file already exist');
      end
   otherwise
      % do nothing
end
  
Fid = fopen(FileName,'w','b');  % machineformat: ieee-be
if (Fid==-1),
   fprintf('Error while attempting to open FITS file for writing\n');
   Flag = 0;
else   
   %--- Write header to file ---
   
   %fwrite(Fid,HeaderBlock,'char');
   
   for Ic=1:1:size(HeaderBlock,1),
     fprintf(Fid,'%-80s',HeaderBlock(Ic,:));
   end
   fprintf(Fid,'%s',PadBlock);

   %--- Write image data ---
   switch DataType
    case {'int8',8}
       fwrite(Fid,Image.','uchar');
    case {'int16',16}
       fwrite(Fid,Image.','int16');
    case {'int32',32}
       fwrite(Fid,Image.','int32');
    case {'int64',64}
       fwrite(Fid,Image.','int64');
    case {'single','float32',-32}
       fwrite(Fid,Image.','single');
    case {'double','float64',-64}
       fwrite(Fid,Image.','double');
    otherwise
       fclose(Fid);
       error('Unknown DataType option');
   end
   fclose(Fid);
end
