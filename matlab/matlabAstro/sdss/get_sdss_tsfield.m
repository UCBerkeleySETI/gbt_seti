function [Links,FitsName,Images,Headers,CatCol]=get_sdss_tsfield(ID,Save,DataRelease);
%------------------------------------------------------------------------------
% get_sdss_tsfield function                                               sdss
% Description: Given an SDSS field ID [Run, Rerun, Camcol, Field], get the
%              link to, and the SDSS field catalog information (tsField)
%              associated with a corrected image in FITS format.
%              Furthermore, read the catalog into a matlab matrix.
%              The program first check if the FITS table exist in
%              current directory and if so, it reads the image from the disk.
%              Note that if nargout>1 then the fits file is retrieved.
% Input  : - Matrix of images ID [Run, ReRun, CamCol, Field].
%            If ReRun is NaN, then the program will look
%            for the most recent rerun in the SDSS archive directory.
%          - Save FITS images to disk {'y' | 'n'}, default is 'gzip'.
%          - Data release {'DRsup'|'DR5','DR6','DRSN','DR7'}, default is 'DR7'.
%            This parameter is obsolete. Always use 'DR7'.
%            If empty use default.
% Output : - Cell array of links to each fits table.
%            Rows for each field, columns for each band.
%          - Cell array of fits tables name.
%          - Cell array tables.
%            Rows for each field, columns for each band.
%            If cell is NaN - image can't retrieved.
%            Each table is stored as a cell array of columns.
%          - Cell array of images header.
%            Rows for each field, columns for each band.
%            Note that keywords are stored in: Header{:,:}.PrimaryData.Keywords
%          - Cell array of all columns in FITS table.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                      June 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Link,FN,Image,Header,CatCol]=get_sdss_tsfield([1339 40 1 11],'y');
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;

SDSS_Server = 'http://das.sdss.org/imaging/';

DefSave        = 'gzip';
DefDataRelease = 'DR7';
if (nargin==1),
   Save         = DefSave;
   DataRelease  = DefDataRelease;
elseif (nargin==2),
   DataRelease  = DefDataRelease;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(DataRelease)),
   DataRelease = DefDataRelease;
end

Nim    = size(ID,1);

Links  = cell(Nim,1);
for Iim=1:1:Nim,
   Run       = ID(Iim,1);
   ReRun     = ID(Iim,2);
   CamCol    = ID(Iim,3);
   Field     = ID(Iim,4);

   if (isnan(ReRun)==1),
      TestURL = sprintf('%s%d/',SDSS_Server,Run);
      Str     = urlread(TestURL);
      Ifolder = findstr(Str,'folder.gif');
      Nfolder = length(Ifolder);
      if (Nfolder==0),
	 error(sprintf('No rerun found in run directory: %s',TestURL));
      else
 	 AllReRun = zeros(Nfolder,1);
	 for Irerun=1:1:Nfolder,
	    Ihref = findstr(Str,'href');          
            Icand = find(Ihref>Ifolder(Irerun));
            Icand = min(Ihref(Icand));
            Splited = regexp(Str(Icand:end),'"','split');
            AllReRun(Irerun) = str2double(Splited{2}(1:end-1));
         end
         ReRun = max(AllReRun);
      end
   end

   switch lower(DataRelease)
    case 'dr7'
       URL = sprintf('%s%d/%d/calibChunks/%d/',SDSS_Server,Run,ReRun,CamCol);
    otherwise
       error('only dr7 release is allowd');
   end


   Retrieve{Iim} = 1;

   ImageName = sprintf('tsField-%06d-%d-%d-%04d.fit',Run,CamCol,ReRun,Field);

   Links{Iim} = sprintf('%s%s',URL,ImageName);

   if (nargout>1),
      FitsName{Iim}= ImageName;

      %--- Check if file is already on disk ---
      if (exist(ImageName,'file')~=0),
         % File is already on local disk
      else
         % file is not on disk - get file
         system(sprintf('wget -q %s',Links{Iim}));

         if (exist(ImageName,'file')==0),
            % Can't retriev image
            Retrieve{Iim} = 0;
         else
            Retrieve{Iim} = 1;
         end
      end 

      switch Retrieve{Iim}
       case 1
          if (nargout>2),
             %--- read image ---
             FitsMat           = fitsread(ImageName,'BinTable');
    
             Images{Iim} = FitsMat;
             if (nargout>3),
                FitsHeader         = fitsinfo(ImageName);
                Headers{Iim} = FitsHeader;
             end
          end
    
          switch Save
           case 'y'
              % do nothing - file is already ungziped
           case 'n'
              % delete FITS image (not gzip any more)
              delete(sprintf('%s',ImageName));
           otherwise
              error('Unknown Save Option');
          end
       case 0
          Headers{Iim} = NaN;
          Images{Iim}  = NaN;
       otherwise
          error('Unknown Retrieve Option');
      end
   end
end

if (nargout>4),
   [CatCol] = get_fitstable_col(FitsName{1});
end



