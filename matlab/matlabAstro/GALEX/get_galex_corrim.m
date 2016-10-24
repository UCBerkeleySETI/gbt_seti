function [Links,FitsName,Images,Headers]=get_galex_corrim(ID,Save,Filters,DataRelease,ImageType)
%------------------------------------------------------------------------------
% get_galex_corrim function                                              GALEX
% Description: Given an GALEX field ID
%              [Vsn, Tilenum, Type, Ow, Prod, Img, Try], get the
%              link to, and the GALEX intensity/rrhr or count
%              image in FITS format.
%              Furthermore, read the corrected image into matlab matrix.
%              The program first check if the FITS image exists in
%              current directory and if so, it reads the image from the disk.
%              Note that if nargout>1 then the fits file is retrieved.
% Input  : - Matrix of images ID [Vsn, Tilenum, Type, Ow, Prod, Img, Try].
%          - Save FITS images to disk {'gzip' | 'y' | 'n'}, default is 'gzip'.
%          - Filters to retrieve {'n' | 'f'}, default is 'nf' (NUV and FUV).
%          - Data release {'GI','GR4','GR5','GR6'}, default is 'GR6'.
%          - Image type (extension {'int' | 'cnt' | 'rrhr'}.
%            Default is 'int'.
% Output : - Cell array of links to each fits image.
%            Rows for each field, columns for each band.
%	     A third dimension is added if the image is divided into
%            segments.
%          - Cell array of fits images name.
%          - Cell array of images matrix.
%            Rows for each field, columns for each band.
%	     A third dimension is added if the image is divided into
%            segments.
%            If cell is NaN - image cannott retrieved.
%          - Cell array of images header.
%            Rows for each field, columns for each band.
%	     A third dimension is added if the image is divided into segments.
%            Note that keywords are stored in: Header{:,:}.PrimaryData.Keywords
% Tested : Matlab 7.5.0
%     By : Ilia Labzovsky                 Apr 2012
% 		   Based on Eran O. Ofek's get_sdss_corrim
%    URL : http://weizmann.ac.il/home/eofek/matlab
% Example: [Link,FN,Image,Header]=get_galex_corrim([1 3000 0 1 1 1 7],'gzip','n');
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;

GALEX_Server = 'http://galex.stsci.edu/data/';

DefSave        = 'gzip';
DefFilters     = 'nf';
DefDataRelease = 'GR6';
DefImageType   = 'int';
if (nargin==1),
	Save         = DefSave;
	Filters      = DefFilters;
	DataRelease  = DefDataRelease;
        ImageType    = DefImageType;
elseif (nargin==2),
	Filters      = DefFilters;
	DataRelease  = DefDataRelease;
        ImageType    = DefImageType;
elseif (nargin==3),
	DataRelease  = DefDataRelease;
        ImageType    = DefImageType;
elseif (nargin==4),
        ImageType    = DefImageType;
elseif (nargin==5),
	% do nothing
else
	error('Illegal number of input arguments');
end

if (isempty(DataRelease)),
	DataRelease = DefDataRelease;
end

Nband  = length(Filters);
Nim    = size(ID,1);

for Iim=1:1:Nim,
	% Read object ID
	Vsn 		= ID(Iim,1);
	TileNum 	= ID(Iim,2);
	Type 		= ID(Iim,3);
	Ow 			= ID(Iim,4);
	Prod	 	= ID(Iim,5);
	Img 		= ID(Iim,6);
	Try 		= ID(Iim,7);
	% Check and process input
	VsnStr = sprintf('%02d-vsn',Vsn);
	TileNumStr = sprintf('%05d',TileNum);
	if (Type==0),
		TypeStr = 's';
	elseif (Type==1),
		TypeStr = 'm';
	else
		error(sprintf('Obs type is not 0(single) or 1(multi)'));
	end
	if (Ow==1),
		OwStr = 'd';
	elseif (Ow~=2),
		OwStr = 'g';
	elseif (Ow~=3),
		OwStr = 'o';
	else
		error(sprintf('Optics wheel is not 1(drct),2(grsm) or 3(opaq)'));
	end
	if (Prod==1),
		ProdStr = sprintf('%02d-main',Prod);
	else
		ProdStr = sprintf('%02d-visits',Prod);
	end
	ImgStr = sprintf('%04d-img',Img);
	TryStr = sprintf('%02d-try',Try);
	
	% Find TileName
	TestURL = sprintf('%s%s/pipe/%s/',GALEX_Server,DataRelease,VsnStr);
	Str     = urlread(TestURL);
	Ifolder = findstr(Str,TileNumStr);
	Nfolder = length(Ifolder);
	TileNameFound = false;
	for Irerun=2:2:Nfolder,
		if (Str(Ifolder(Irerun)-1)=='>'),
			TileName = Str(Ifolder(Irerun-1)+6:Ifolder(Irerun)-4);
			TileNameFound = true;
			break;
		end
	end
	if (TileNameFound==false),
		error(sprintf('TileNum %s was not found in run directory: %s',TileNumStr,TestURL));
	end
	% Find ImgStr (in vsn=2 the site has a different img number from what i get from sql)
	TestURL = sprintf('%s%s/pipe/%s/%s-%s/%s/%s/',GALEX_Server,DataRelease,VsnStr,TileNumStr,TileName,OwStr,ProdStr);
	Str     = urlread(TestURL);
	Ifolder = findstr(Str,ImgStr);
	Nfolder = length(Ifolder);
	if (Nfolder==0),
		Ifolder = findstr(Str,'-img');
		Nfolder = length(Ifolder);
		if (Nfolder>0 && Str(Ifolder(Irerun-1)+6)=='>'),
			ImgStr = Str(Ifolder(Irerun-1)+7:Ifolder(Irerun)+3);
		else
			error(sprintf('img was not found in run directory: %s',TestURL));
		end
	end
	% Find URL
	URL = sprintf('%s%s/pipe/%s/%s-%s/%s/%s/%s/%s/',GALEX_Server,DataRelease,VsnStr,TileNumStr,TileName,OwStr,ProdStr,ImgStr,TryStr);
	Str = urlread(URL);
	
	for Iband=1:1:Nband,
		Retrieve{Iim,Iband} = 1;
   
		% Find ImageName
		ImageNameEnd = sprintf('-%s%s-%s.fits.gz',Filters(Iband),OwStr,lower(ImageType));
		Ifolder = findstr(Str,ImageNameEnd);
		Nfolder = length(Ifolder);
		
		for Irerun=2:2:Nfolder,
			if (Str(Ifolder(Irerun-1)+16)=='>'),
				ImageName = Str(Ifolder(Irerun-1)+17:Ifolder(Irerun)+14);
			else
				error(sprintf('TileNum %s was not found in run directory: %s',TileNumStr,TestURL));
			end
			Links{Iim,Iband,Irerun/2} = sprintf('%s%s',URL,ImageName);
		
			if (nargout>1),
				switch Save
				case 'gzip'
					FitsName{Iim,Iband,Irerun/2}= ImageName;
				otherwise
					FitsName{Iim,Iband,Irerun/2}= ImageName(1:end-3);
				end

				%--- Check if file is already on disk ---
				if (exist(ImageName,'file')~=0 | exist(ImageName(1:end-3),'file')~=0),
					% File is already on local disk
					if (exist(ImageName,'file')==0),
						FileIsGZIP = 0;
					else
						FileIsGZIP = 1;
					end
				else
					% file is not on disk - get file
					system(sprintf('wget -q %s',Links{Iim,Iband,Irerun/2}));


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
							system(sprintf('gzip -d %s',ImageName));
						case 0 
							% do nothing
						otherwise
							error('Unknown FileIsGZIP Option');
						end 
						FitsMat = fitsread(ImageName(1:end-3));
		
						Images{Iim,Iband,Irerun/2} = FitsMat;
						if (nargout>3),
							FitsHeader         = fitsinfo(ImageName(1:end-3));
							Headers{Iim,Iband,Irerun/2} = FitsHeader;
						end
					end
				 
					switch Save
					case 'gzip'
						% gzip file
						system(sprintf('gzip %s',ImageName(1:end-3)));
					case 'y'
						% do nothing - file is already ungziped
					case 'n'
						% delete FITS image (not gzip any more)
						%system(sprintf('rm %s',ImageName(1:end-3)));
						delete(sprintf('%s',ImageName(1:end-3)));
					otherwise
						error('Unknown Save Option');
					end
				case 0
					Headers{Iim,Iband,Irerun/2} = NaN;
					Images{Iim,Iband,Irerun/2}  = NaN;
				otherwise
					error('Unknown Retrieve Option');
				end
			end
		end
	end
end
