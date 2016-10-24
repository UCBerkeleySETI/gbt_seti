function [BiasCell,NonBiasCell]=identify_bias_fits(ImInput,varargin)
%-----------------------------------------------------------------------------
% identify_bias_fits function                                         ImBasic
% Description: Given a set of images, look for all the bias images.
%              This script can be used also to identify errornous bias
%              images.
% Input  : - List of input images (see create_list.m for details).
%            Default is '*.fits'.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'ImTypeKey'  - Image type keywords in header,
%                           default is {'IMTYPE','IMAGETYP','TYPE','IMGTYP'}.
%                           If more than one keyword is given then will
%                           inspect only the first existing keyword.
%                           If empty then will not attempt to look for
%                           bias using the IMTYPE header keyword.
%            'ImTypeVal'  - Image type value to search for,
%                           default is 'bias'.
%                           This can be a string or cell array of strings.
%            'ByName'     - If not empty matrix then will attempt to look 
%                           for bias images by looking for a specific sub string,
%                           given following this keyword, in the image name.
%                           Default is 'bias'.
%            'NotBias'    - A cell array of strings containing image type
%                           values which are not bias images.
%                           If empty matrix, then no such strings.
%                           Default is {'science','object','flat','twflat','twilight','led','test','focus'}
%            'NotFound'   - A parameter specify what to do in case images
%                           with the specifim IMTYPE are not found.
%                           Option are:
%                           'stop' - Stop searching and return an empty list.
%                           'stat' - Attemt to look for biases using image
%                                    statistical properties.
%                                    Default is 'stat'.
%            'MaxCountRange' - The allowed maximum range of the median
%                           of bias images. Default is 30.
%            'Verbose'    - Print information regarding the processing {'y' | 'n'}.
%                           Default is 'y'.
%          - Cell aray of all bias images.
%          - Cell array of all non-bias images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%-----------------------------------------------------------------------------

Def.ImInput  = '*.fits';
if (nargin==0),
   ImInput   = Def.ImInput;
end

DefV.ImTypeKey     = {'IMTYPE','IMAGETYP','TYPE','IMGTYP'};
DefV.ImTypeVal     = 'bias';
DefV.ByName        = 'bias';
DefV.NotFound      = 'stat';
DefV.NotBias       = {'science','object','flat','twflat','twilight','led','test','focus'};
DefV.Verbose       = 'y';
DefV.MaxCountRange = 30;

InPar = set_varargin_keyval(DefV,'n','def',varargin{:});


if (~iscell(InPar.ImTypeKey)),
   InPar.ImTypeKey = {InPar.ImTypeKey};
end

if (~iscell(InPar.ImTypeVal)),
   InPar.ImTypeVal = {InPar.ImTypeVal};
end

% Generate list from ImInput:
[~,ImInputCell] = create_list(ImInput,NaN);
Nim = length(ImInputCell);

BiasFlag = zeros(Nim,1).*NaN;   % NaN - unknown; 1 - Bias image; 0 - not a bias image
if (isempty(InPar.ImTypeKey)),
   % skip the attempt to look for bias images by the IMTYPE header keyword
else
   % attempt to look for bias images by the IMTYPE header keyword
   for Iim=1:1:Nim,
      KeyVal  = get_fits_keyword(ImInputCell{Iim},InPar.ImTypeKey);

      FlagNaN = isnan_cell(KeyVal);
      Ikey    = find(FlagNaN==0,1,'first');

      if (isempty(Ikey)),
         % IMTYPE keyword is not present in header
         BiasFlag(Iim) = NaN;
      else
         % IMTYPE keyword is present in header
         if (ischar(KeyVal{Ikey})),

            %[strcmpi(strtrim(KeyVal{Ikey}),InPar.ImTypeVal), ~prod(double(isempty_cell(strfind(lower(InPar.ImTypeVal),lower(strtrim(KeyVal{Ikey}))))))]
            if (~prod(double(isempty_cell(strfind(lower(InPar.ImTypeVal),lower(strtrim(KeyVal{Ikey}))))))),
	       %if (strcmpi(strtrim(KeyVal{Ikey}),InPar.ImTypeVal)),
               % Bias image found using the IMTYPE keyword
	       BiasFlag(Iim) = 1;
            else
	        if (isempty(InPar.NotBias)),
                   % user didn't specified NotBias keywords
                   % assume the keywod type is unknown
		   BiasFlag(Iim) = NaN;
               else
                  % user specified NotBias keywords
		  if (sum(strcmpi(strtrim(KeyVal{Ikey}),InPar.NotBias))>0),
                     % IMTYPE value is consistent with a non-bias image
                     % listed in NotBias.
		     BiasFlag(Iim) = 0;
                  else
                     % image type is unknown
		     BiasFlag(Iim) = NaN;
                  end
               end
            end
         else
            % IMTYPE value is not a string
	    BiasFlag(Iim) = NaN;
         end
      end
   end
end

Nbias = length(find(BiasFlag==1));

switch lower(InPar.Verbose)
 case 'y'
    fprintf('identify_bias_fits.m found %d bias images based on the IMTYPE header keyword\n',Nbias);
 otherwise
    % do nothing
end


%for Iim=1:1:Nim,
%   fprintf('%s   %d\n',ImInputCell{Iim},BiasFlag(Iim))
%end

%--- look for bias images by looking for the string 'bias' in the image names ---
if (Nbias==0),
   if (~isempty(InPar.ByName)),
      IndBias = find(isempty_cell(strfind(ImInputCell,InPar.ByName))==0);
      BiasFlag(IndBias) = 1;
   end
end

Nbias = length(find(BiasFlag==1));


switch lower(InPar.Verbose)
 case 'y'
    fprintf('identify_bias_fits.m found %d bias images based on the IMTYPE header keyword and image name\n',Nbias);
 otherwise
    % do nothing
end


if (Nbias==0),   
   switch lower(InPar.NotFound)
    case 'stop'
       % do nothing
    case 'stat'
       %--- attempt to identify bias images using image statistics ---
       [Stat] = imstat_fits(ImInputCell,'ImType','fits');

       for Iim=1:1:Nim,
         Stat(Iim).RobStD = Stat(Iim).Percentile(1,2)-Stat(Iim).Median;
       end
    
       Flag1 = (abs([Stat(:).RobStD]./[Stat(:).StD])<1.1);
    
       MinMedian = min([Stat(:).Median]);
    
       Flag2 = ([Stat(:).Median]>=MinMedian & [Stat(:).Median]<(MinMedian+InPar.MaxCountRange));

       % select Bias images
       BiasFlag = Flag1 & Flag2;

    otherwise
       error('Unknown NotFound option');
   end
end

Nbias = length(find(BiasFlag==1));

switch lower(InPar.Verbose)
 case 'y'
    fprintf('identify_bias_fits.m found %d bias images based on the IMTYPE header keyword and image name and image statistics\n',Nbias);
 otherwise
    % do nothing
end

%--- Select the best bias image candidates ---
Icand    = find(BiasFlag==1);
BiasCell = ImInputCell(Icand);

%--- Select non-bias images ---
Icand       = find(BiasFlag~=1);
NonBiasCell = ImInputCell(Icand);
