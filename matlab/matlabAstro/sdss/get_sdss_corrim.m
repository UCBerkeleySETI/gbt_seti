function [Links,FitsName,Images,Headers]=get_sdss_corrim(ID,Save,Filters,DataRelease)
%--------------------------------------------------------------------------
% get_sdss_corrim function                                            sdss
% Description: Given an SDSS field ID [Run, Rerun, Camcol, Field], get the
%              link to, and the SDSS corrected image in FITS format.
%              Furthermore, read the corrected image into matlab matrix.
%              The program first check if the FITS image is exist in
%              current directory and if so, it reads the image from the disk.
%              Note that if nargout>1 then the fits file is retrieved.
% Input  : - Matrix of images ID [Run, ReRun, CamCol, Field].
%            If ReRun is NaN, then the program will look
%            for the most recent rerun in the SDSS archive directory.
%          - Save FITS images to disk {'gzip' | 'y' | 'n'}, default is 'gzip'.
%          - Filters to retrieve, default is 'ugriz' (all filters).
%          - Data release {'DRsup'|'DR5','DR6','DRSN','DR7'}, default is 'DR7'.
%            This parameter is obsolete. Always use 'DR7'.
%            If empty use default.
% Output : - Cell array of links to each fits image.
%            Rows for each field, columns for each band.
%          - Cell array of fits images name.
%          - Cell array of images matrix.
%            Rows for each field, columns for each band.
%            If cell is NaN - image can't retrieved.
%          - Cell array of images header.
%            Rows for each field, columns for each band.
%            Note that keywords are stored in: Header{:,:}.PrimaryData.Keywords
% Tested : Matlab 7.0
%     By : Eran O. Ofek                      June 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Link,FN,Image,Header]=get_sdss_corrim([1339 40 1 11],'gzip','ug');
% Reliable: 2
%--------------------------------------------------------------------------
RAD = 180./pi;

SDSS_Server = 'http://das.sdss.org/imaging/';

DefSave        = 'gzip';
DefFilters     = 'ugriz';
DefDataRelease = 'DR9';
if (nargin==1),
   Save         = DefSave;
   Filters      = DefFilters;
   DataRelease  = DefDataRelease;
elseif (nargin==2),
   Filters      = DefFilters;
   DataRelease  = DefDataRelease;
elseif (nargin==3),
   DataRelease  = DefDataRelease;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(DataRelease)),
   DataRelease = DefDataRelease;
end

Nband  = length(Filters); 
Nim    = size(ID,1);

Links  = cell(Nim,Nband);
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

            % look for "corr/" directory in rerun directory:
            TestURL = sprintf('%s%d/%d/',SDSS_Server,Run,AllReRun(Irerun));
            StrRe    = urlread(TestURL);
            if (isempty(findstr(StrRe,'corr/'))),
               AllReRun(Irerun) = NaN;
            end

         end
         ReRun = max(AllReRun);
      end
   end

   switch lower(DataRelease)
    case 'dr7'
       ImRootName = 'fpC';
       ImComp     = 'fit.gz';
	   Comp       = 'gzip';
	   CompLength = 3;
       URL = sprintf('%s%d/%d/corr/%d/',SDSS_Server,Run,ReRun,CamCol);
    case 'dr9'
       ImRootName = 'frame';
       ImComp     = 'fits.bz2';
	   Comp       = 'bzip2';
	   CompLength = 4;
       URL = sprintf('http://data.sdss3.org/sas/dr9/boss/photoObj/frames/%d/%d/%d/',ReRun,Run,CamCol);
    otherwise
       error('only dr7/dr9 release is allowd');
   end

   for Iband=1:1:Nband,
      Retrieve{Iim,Iband} = 1;

	  switch lower(DataRelease)
		case 'dr7'
		   ImageName = sprintf('%s-%06d-%c%d-%04d.%s',ImRootName,Run,Filters(Iband),CamCol,Field,ImComp);
		case 'dr9'
		   ImageName = sprintf('%s-%c-%06d-%d-%04d.%s',ImRootName,Filters(Iband),Run,CamCol,Field,ImComp);
		otherwise
		   error('only dr7/dr9 release is allowd');
	   end

      Links{Iim,Iband} = sprintf('%s%s',URL,ImageName);

      if (nargout>1),
         switch Save
          case 'gzip'
             FitsName{Iim,Iband}= ImageName;
          otherwise
             FitsName{Iim,Iband}= ImageName(1:end-CompLength);
         end
if (nargout>2),
         %--- Check if file is already on disk ---
         if (exist(ImageName,'file')~=0 | exist(ImageName(1:end-CompLength),'file')~=0),
            % File is already on local disk
	    if (exist(ImageName,'file')==0),
               FileIsGZIP = 0;
            else
               FileIsGZIP = 1;
            end
	 else
            % file is not on disk - get file
            system(sprintf('wget -q -nc %s',Links{Iim,Iband}));
            %    (sprintf('wget -q -nc %s',Links{Iim,Iband}))

            if (exist(ImageName,'file')==0),
                % Can't retriev image
                Retrieve{Iim,Iband} = 0;
            else
                Retrieve{Iim,Iband} = 1;
            end
            FileIsGZIP = 1;
         end 

         switch Retrieve{Iim,Iband}
          case 1
             if (nargout>2),
                %--- read image ---
                switch FileIsGZIP
                 case 1
                    system(sprintf('%s -d %s',Comp,ImageName));
                 case 0 
                    % do nothing
                 otherwise
                    error('Unknown FileIsGZIP Option');
                end 
                FitsMat           = fitsread(ImageName(1:end-CompLength));
    
                Images{Iim,Iband} = FitsMat;
                if (nargout>3),
                   FitsHeader         = fitsinfo(ImageName(1:end-CompLength));
                   Headers{Iim,Iband} = FitsHeader;
                end
             end
    
             switch Save
              case 'gzip'
                 % gzip file
		 system(sprintf('%s %s',Comp,ImageName(1:end-CompLength)));
              case 'y'
                 % do nothing - file is already ungziped
              case 'n'
                 % delete FITS image (not gzip any more)
                 %system(sprintf('rm %s',ImageName(1:end-CompLength)));
                 delete(sprintf('%s',ImageName(1:end-CompLength)));
              otherwise
                 error('Unknown Save Option');
             end
          case 0
             Headers{Iim,Iband} = NaN;
             Images{Iim,Iband}  = NaN;
          otherwise
             error('Unknown Retrieve Option');
         end
      end
   end
end
end