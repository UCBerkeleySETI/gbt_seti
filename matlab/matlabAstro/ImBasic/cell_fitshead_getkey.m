function [NewCellHead,Lines]=cell_fitshead_getkey(CellHead,Keys,NotExist,Multiple)
%--------------------------------------------------------------------------
% cell_fitshead_getkey function                                    ImBasic
% Description: A utility program to get a specific keywords, values and
%              comments from a cell array containing A FITS header
%              information.
%              The FITS header cell array contains an arbitrary number
%              of rows and 3 columns, where the columns are:
%              {keyword_name, keyword_val, comment}.
%              The comment column is optional.
% Input  : - Cell array containing the FITS header information.
%          - A string containing a keyword to look in the header
%            or a cell array of keywords (case insensitive).
%          - A parameter to control the behaviour when a specific keyword
%            is not found.
%            'Ignore' - ignore missing parameters and in that case the
%                       length of NewCellHead will be shorter than the
%                       length of (CellHead). Default.
%            'NaN'    - Replace missing keywords by NaNs.
%                       In that case the Lines vector and the NewCellHead
%                       will contain NaNs
%          - A flag indicating what to do if there are multiple apperances
%            of a specific keyword. Options are: {'all','first','last'}.
%            Default is 'last'.
% Output : - The header cell array with only the requested lines.
%          - Indices of requested lines in header.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [NewCellHead,Lines]=cell_fitshead_getkey(CellHead,Keys,NotExist);
% Reliable: 2
%--------------------------------------------------------------------------

Def.NotExist = 'Ignore';
Def.Multiple = 'last';
if (nargin==2),
   NotExist = Def.NotExist;
   Multiple = Def.Multiple;
elseif (nargin==3),
    Multiple = Def.Multiple;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (ischar(Keys)),
   Keys = {Keys};
end

Nkeys = length(Keys);
Lines = zeros(0,1);
for Ikeys=1:1:Nkeys,
    switch lower(Multiple)
        case 'all'
            Ifound = find(strcmpi(CellHead(:,1),Keys{Ikeys})==1);
        otherwise
            Ifound = find(strcmpi(CellHead(:,1),Keys{Ikeys})==1,1,Multiple);
    end
    

   Lines  = [Lines; Ifound];
   
   if (isempty(Ifound)),
      switch lower(NotExist)
       case 'nan'
          Lines = [Lines; NaN];
       otherwise
          % do nothing
      end
   end
end

if (sum(isnan(Lines))>0),
   Inan = find(isnan(Lines));
   Lines(Inan) = 1;
   NewCellHead = CellHead(Lines,:);
   for In=1:1:length(Inan),
      NewCellHead(Inan(In),:) = {NaN NaN NaN};
   end
else
   NewCellHead = CellHead(Lines,:);
end
