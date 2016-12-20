function [KeywordVal,KeywordS]=get_fits_keyword(FitsName,Keys,TableInd,TableName,OnlyVal)
%-------------------------------------------------------------------------
% get_fits_keyword function                                       ImBasic
% Description: Get list of user selected keywords value from the
%              header of a single FITS file.
%              See also: mget_fits_keyword.m
% Input  : - FITS file name (string)
%            or a the structure containing the image header
%            (as returned by fitsinfo function).
%            or a cell array containing the keywords.
%            The cell array may contain 3 columns {keyword_name,
%            keyword_val, comment}.
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
% Output : - Cell array of corresponding keyword values.
%          - Structure in which each filed is the requested keyword.
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

if (ischar(FitsName)),
   % read FITS image
   try
       Info       = fitsinfo(FitsName);
   catch
       Info       = [];
   end
elseif (iscell(FitsName)),
   InfoKey    = FitsName;
elseif (isstruct(FitsName)),
   % keyword structure
   Info       = FitsName;
else
   error('Illegal class for FitsName input');
end

if (isempty(TableInd)),
    TableInd = NaN;
end

if (~iscell(FitsName)),
   if (isempty(TableName)),
      if (isnan(TableInd)),
         TableName = 'PrimaryData';
      else 
         TableName = 'BinaryTable';
      end
   end

   if (isnan(TableInd) || isempty(TableInd)),
      TableInd = 1;
   end

   if (isempty(Info)),
       InfoKey = [];
   else
       InfoKey = getfield(getfield(Info,TableName,{TableInd}),'Keywords');
   end
end

if (isempty(Keys) || isempty(InfoKey)),
   NewCellHead = InfoKey;
else
   [NewCellHead,Lines]=cell_fitshead_getkey(InfoKey,Keys,'NaN');
end
   
if (isempty(NewCellHead)),
    KeywordVal = [];
    KeywordS   = [];
else
    switch lower(OnlyVal)
     case 'y'
        KeywordVal = NewCellHead(:,2);

        Nk = length(KeywordVal);
        for Ik=1:1:Nk,
           if (~isnumeric(KeywordVal{Ik})),
              NumVal = str2double(KeywordVal{Ik});
              if (~isnan(NumVal)),
                 KeywordVal{Ik} = NumVal;
              end
           end
        end

     case 'n'
        KeywordVal = NewCellHead;

     otherwise
        error('Unknown OnlyVal option');
    end


    if (nargout>1),
       switch lower(OnlyVal)
           case 'n'
               KeywordV = NewCellHead(:,2);

               Nk = length(KeywordV);
               for Ik=1:1:Nk,
                  if (~isnumeric(KeywordV{Ik})),
                     NumVal = str2double(KeywordV{Ik});
                     if (~isnan(NumVal)),
                        KeywordV{Ik} = NumVal;
                     end
                  end
               end
           case 'y'
               KeywordV = KeywordVal;
           otherwise
               error('Unknown OnlyVal option');
       end
       Ncol       = length(Keys);
       KeywordS  = cell2struct(KeywordV,Keys,1);
    end
end
