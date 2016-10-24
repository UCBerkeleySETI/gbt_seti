function [FlatCell,NonFlatCell,SatFlatCell]=identify_flat_fits(ImInput,varargin);
%-----------------------------------------------------------------------------
% identify_flat_fits function                                         ImBasic
% Description: Given a set of images, look for all the flat images.
%              The program also check if some of the flat images are
%              saturated and if so reject them from the list of flat images.
% Input  : - List of input images (see create_list.m for details).
%            Default is '*.fits'. If empty matrix use default.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'ImTypeKey'  - Image type keywords in header,
%                           default is {'IMTYPE','IMAGETYP','TYPE','IMGTYP'}.
%                           If more than one keyword is given then will
%                           inspect only the first existing keyword.
%                           If empty then will not attempt to look for
%                           flat using the IMTYPE header keyword.
%            'ImTypeVal'  - Image type value to search for,
%                           default is {'flat','flatfield','ff','twflat','twilight'}.
%                           This can be a string or cell array of strings.
%            'ByName'     - If not empty matrix then will attempt to look 
%                           for flat images by looking for a specific string,
%                           given following this keyword, in the image name.
%                           Default is 'flat'.
%            'NotFlat'    - A cell array of strings containing image type
%                           values which are not flat images.
%                           If empty matrix, then no such strings.
%                           Default is
%                           {'science','object','bias','led','test','focus'}
%            'OnlyCheck'  - {'y' | 'n'}, default is 'n'. If 'y' then will
%                           skip the flat image search and will assume
%                           that all the images in the input are flat images
%                           and check for saturation (if CheckSat='y').
%            'CheckSat'   - Check if Flat image is saturated {'y' | 'n'},
%                           default is 'y'.
%            'SatLevel'   - Saturation level [adu].
%                           If CheckStat='y' then program will discard images
%                           in which the fraction of saturated pixels is larger
%                           than 'SatFrac'.
%                           Default is 30000.
%            'SatFrac'    - Maximum allowed fraction of saturated pixels.
%                           Default is 0.02. This is used only if CheckSat='y'.
%            'SatKey'     - A string or cell array of strings containing
%                           header keyword which store the saturation level
%                           of the CCD. Default is empty matrix.
%                           If empty matrix than do not use header keywords.
%                           If given this parameter overrides the 'SatLevel'
%                           parameter.
%                           If several keywords are given than will select the
%                           first keyword in the cell array which exist in header.
%                           If keyword not avilable in header than will use the
%                           value in 'SatLevel'.
%            'Verbose'    - Print processing and progress information {'y' | 'n'}.
%                           Default is 'y'.
% Output : - Cell array of all flat images.
%            If CheckSat='y' then saturated flat images will be removed
%            from this list.
%          - Cell array of all non-flat images.
%          - Cell array of flat images that are saturated.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [FlatCell,NonFlatCell,SatFlatCell]=identify_flat_fits;
%          [FlatCell,NonFlatCell,SatFlatCell]=identify_flat_fits('r00*.fits');
% Reliable: 2
%-----------------------------------------------------------------------------

FlatCell    = {};
NonFlatCell = {};
SatFlatCell = {};

Def.ImInput  = '*.fits';
if (nargin==0),
   ImInput   = Def.ImInput;
end

DefV.ImTypeKey     = {'IMTYPE','IMAGETYP','TYPE','IMGTYP'};
DefV.ImTypeVal     = {'flat','flatfield','ff','twflat','twilight'};
DefV.ByName        = 'flat';
DefV.NotFlat       = {'science','object','bias','led','test','focus'};
DefV.OnlyCheck     = 'n';
DefV.CheckSat      = 'y';
DefV.SatLevel      = 30000;
DefV.SatFrac       = 0.02;
DefV.SatKey        = [];
DefV.Verbose       = 'y';

InPar = set_varargin_keyval(DefV,'n','def',varargin{:});

if (isempty(ImInput)),
   ImInput   = Def.ImInput;
end

if (~iscell(InPar.ImTypeKey)),
   InPar.ImTypeKey = {InPar.ImTypeKey};
end

if (~iscell(InPar.ImTypeVal)),
   InPar.ImTypeVal = {InPar.ImTypeVal};
end

% Generate list from ImInput:
[~,ImInputCell] = create_list(ImInput,NaN);
Nim = length(ImInputCell);

switch lower(InPar.OnlyCheck)
 case 'n'
    % search for flat images before checking for saturation

    FlatFlag = zeros(Nim,1).*NaN;   % NaN - unknown; 1 - flat image; 0 - not a flat image
    if (isempty(InPar.ImTypeKey)),
       % skip the attempt to look for flat images by the IMTYPE header keyword
    else
       % attempt to look for flat images by the IMTYPE header keyword
       for Iim=1:1:Nim,
          KeyVal  = get_fits_keyword(ImInputCell{Iim},InPar.ImTypeKey);
          FlagNaN = isnan_cell(KeyVal);
          Ikey    = find(FlagNaN==0,1,'first');
          
          if (isempty(Ikey)),
             % IMTYPE keyword is not present in header
             FlatFlag(Iim) = NaN;
          else
             % IMTYPE keyword is present in header
             if (ischar(KeyVal{Ikey})),
                if (~prod(double(isempty_cell(strfind(lower(InPar.ImTypeVal),lower(strtrim(KeyVal{Ikey}))))))),
                   % flat image found using the IMTYPE keyword
    	            FlatFlag(Iim) = 1;
                else
    	           if (isempty(InPar.NotFlat)),
                      % user didn't specified NotFlat keywords
                      % assume the keywod type is unknown
    		      FlatFlag(Iim) = NaN;
                   else
                      % user specified NotFlat keywords
    		      if (sum(strcmpi(strtrim(KeyVal{Ikey}),InPar.NotFlat))>0),
                         % IMTYPE value is consistent with a non-flat image
                         % listed in NotFlat.
    		         FlatFlag(Iim) = 0;
                      else
                         % image type is unknown
    		         FlatFlag(Iim) = NaN;
                      end
                   end
                end
             else
                % IMTYPE value is not a string
                FlatFlag(Iim) = NaN;
             end
          end
       end
    end
    
    Nflat = length(find(FlatFlag==1));
    
    
    %for Iim=1:1:Nim,
    %   fprintf('%s   %d\n',ImInputCell{Iim},FlatFlag(Iim))
    %end
    
    %--- look for flat images by looking for the string 'flat' in the image names ---
    if (Nflat==0),
       if (~isempty(InPar.ByName)),
          IndFlat = find(isempty_cell(strfind(ImInputCell,InPar.ByName))==0);
          FlatFlag(IndFlat) = 1;
       end
    end

    switch lower(InPar.Verbose)
     case 'y'
        fprintf('identify_flat_fits.m found %d Flat images based on the IMTYPE header keyword\n',Nflat);
     otherwise
        % do nothing
    end

 case 'y'
    % Assume the input list contains only flat images
    FlatFlag = ones(Nim,1);   % NaN - unknown; 1 - flat image; 0 - not a flat image
 otherwise
    error('Unknown OnlyCheck option');
end


%--- Select the best flat image candidates ---
Icand    = find(FlatFlag==1);
FlatCell = ImInputCell(Icand);
Nflat    = length(FlatCell);

%--- Select non-flat images ---
Icand       = find(FlatFlag~=1);
NonFlatCell = ImInputCell(Icand);



switch lower(InPar.CheckSat)
 case 'y'

    if (~isempty(InPar.SatKey)),
       % Read Saturation level from header
       if (~iscell(InPar.SatKey)),
          InPar.SatKey = {InPar.SatKey};
       end

       SatLevelKey = zeros(Nflat,1).*NaN;
       for Iflat=1:1:Nflat,
          % get saturation level from header

          KeySatVal = get_fits_keyword(FlatCell{Iflat},InPar.SatKey);
          Ikey = find(isnan_cell(KeySatVal)==0,1);
          if (isempty(Ikey)),
             % if keyword not available use StLevel value
             SatLevelKey(Iflat) = InPar.SatLevel;
          else
             if (ischar(KeySatVal{Ikey})),
                SatLevelKey(Iflat) = str2double(KeySatVal{Ikey});
             else
                SatLevelKey(Iflat) = KeySatVal{Ikey};
             end
          end
       end
       % replace all the NaN (not found in header) by SatLevel value
       SatLevelKey(find(isnan(SatLevelKey))) = InPar.SatLevel;
    else
       SatLevelKey = InPar.SatLevel.*ones(Nflat,1);
    end

    IsatFlat         = 0;
    Iflat_satFlat    = zeros(0,1);
    Iflat_NotSatFlat = zeros(0,1);
    for Iflat=1:1:Nflat,
       % for each flat image - check for saturation

       FlatImage = fitsread(FlatCell{Iflat});
       if (length(find(FlatImage>SatLevelKey(Iflat)))./prod(size(FlatImage))>InPar.SatFrac),
          % Flat image is probably saturated

          IsatFlat = IsatFlat + 1;
          SatFlatCell{IsatFlat} = FlatCell{Iflat};
          Iflat_satFlat = [Iflat_satFlat; Iflat]; % index of saturated Flats among the flat images ("Icand")

          switch lower(InPar.Verbose)
           case 'y'
              fprintf('Image %s has %6.4f pixels above level of %f - image is probably saturated\n',...
                      FlatCell{Iflat},...
                      length(find(FlatImage>SatLevelKey(Iflat)))./prod(size(FlatImage)),...
                      SatLevelKey(Iflat));
           otherwise
              % do nothing
          end


       else
          % Flat image is probably not saturated
          Iflat_NotSatFlat = [Iflat_NotSatFlat; Iflat];
       end
    end

    % SatFlatCell contains list of saturated flat images
    % FlatCell: select only non-saturated images
    FlatCell = FlatCell(Iflat_NotSatFlat);
         
 case 'n'
    % do nothinh
    SatFlatCell = {};
 otherwise
    error('Unknown CheckSat option');
end

switch lower(InPar.Verbose)
 case 'y'
    fprintf('identify_flat_fits.m found %d saturated Flat images\n',length(SatFlatCell));
 otherwise
    % do nothing
end



