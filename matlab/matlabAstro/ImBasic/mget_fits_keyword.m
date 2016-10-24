function [KeywordVal,KeywordS]=mget_fits_keyword(FitsName,Keys,TableInd,TableName,OnlyVal);
%-------------------------------------------------------------------------
% mget_fits_keyword function                                      ImBasic
% Description: Get list of user selected keywords value from the
%              headers of multiple FITS files.
%              See also get_fits_keyword.m
% Input  : - List of FITS files (see create_list.m for details).
%          - Cell array of keywords to search in header.
%            If empty matrix the return all keywords.
%          - In case the header data unit is a vector containing several
%            extension, then you can specify the extension name.
%            Default is NaN.
%          - String containing the name of the header data unit
%            If TableInd is NaN or empty than default is 'PrimaryData',
%            else default is BinaryTable(TableInd).
%          - Return only keyword value or the entire line {key, val,
%            comment}. Options are {'y' | 'n'}. Default is 'y', return only
%            value.
%            'y' will also attempt to convert strings to doubles.
% Output : - Cell array of cell arrays of corresponding keyword values.
%            E.g., Key{ImageInd}{KeywordInd}.
%          - Structure array in which each field is named after the
%            the keyword nane and it contains the keyword value.
%            E.g., Key(ImageInd).KeywordName = KeywordValue
% Tested : Matlab 7.0
%     By : Eran O. Ofek                       May 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: % Read values for filter and exptime from PrimaryData table:
%          KeywordVal=get_fits_keyword('File.fits',{'FILTER','EXPTIME'});
%          % Read values for filter from BinaryTable:
%          KeywordVal=get_fits_keyword('File.fits',{'FILTER'},1);
%          % or
%          KeywordVal=get_fits_keyword('File.fits',{'FILTER'},[],'BinaryTable');
%          % Read value from specific header data unit called 'Image':
%          KeywordVal=get_fits_keyword('File.fits',{'FILTER'},[],'Image');
% Reliable: 2
%-------------------------------------------------------------------------
Def.TableInd  = NaN;
Def.TableName = [];
Def.OnlyVal   = 'y';
if (nargin==2),
   TableInd   = Def.TableInd;
   TableName  = Def.TableName;
   OnlyVal    = Def.OnlyVal;
elseif (nargin==3),
   TableName  = Def.TableName;
   OnlyVal    = Def.OnlyVal;
elseif (nargin==4),
   OnlyVal    = Def.OnlyVal;
elseif (nargin==5),
   % do nothing
else
   error('Illegal number of input arguments');
end

[~,ListFileCell] = create_list(FitsName,NaN);

Nim = length(ListFileCell);
KeywordVal = cell(Nim,1);
KeywordS   = [];
for Iim=1:1:Nim,
    
   %ListFileCell{Iim}
   Val = get_fits_keyword(ListFileCell{Iim},Keys,TableInd,TableName,OnlyVal);
   
   % try to convert to double
   
   KeywordVal{Iim} = Val;
   if (nargout>1),
      for Ik=1:1:length(Keys),
          if (isempty(Val{Ik})),
              Val{Ik} = NaN;
          end
          KeywordS(Iim).(Keys{Ik}) = Val{Ik};
      end
   end
end

