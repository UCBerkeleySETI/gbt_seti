function [CCDSEC]=get_ccdsec_fits(ImageName,Keyword)
%--------------------------------------------------------------------------
% get_ccdsec_fits function                                         ImBasic
% Description: Get an parse CCD section keyword value from a list
%              of images.
% Input  : - List of images (see create_list.m for details).
%          - String containing CCD section keyword to get and parse.
%            Alternatively this can be a cell array of CCDSEC keywords.
% Output : - Matrix of [Xmin Xmax Ymin Ymax] parsed from the CCD section
%            keyword value. Line per image.
%            If the input was a cell array of CCDSEC keywords than this
%            will be a cell array in which each cell corresponds to one
%            image and each line in each cell corresponds to one keyword.
% Tested : - Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%--------------------------------------------------------------------------

[~,InputCell] = create_list(ImageName,NaN);

if (iscell(Keyword)),
   CellKey = 1;
   Nkey    = length(Keyword);
else
   CellKey = 0;
   Nkey    = 1;
   Keyword = {Keyword};
end

Nim = length(InputCell);
CCDSEC = zeros(Nim,4);
for Iim=1:1:Nim,
   CCDSEC_Val = get_fits_keyword(InputCell{Iim},Keyword);
   for Ikey=1:1:Nkey,

      if (isnan(CCDSEC_Val{Ikey})),
         error('can not find CCDSEC keyword in image header');
      else
         Splitted   = regexp(CCDSEC_Val{Ikey},'[:\[\],]','split');
 
         Xmin       = str2double(Splitted{2});
         Xmax       = str2double(Splitted{3});
         Ymin       = str2double(Splitted{4});
         Ymax       = str2double(Splitted{5});
      end

      switch CellKey
       case 0
          % single keyword was specified - return a matrix      
          CCDSEC(Iim,:) = [Xmin, Xmax, Ymin, Ymax];
       case 1
          % multiple keyword were requested - return a cell array
          CCDSEC{Iim}(Ikey,:) = [Xmin, Xmax, Ymin, Ymax];
       otherwise
          error('Unknown CellKey option');
      end
   end
end
