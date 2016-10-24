function [Stat,Sim]=sim_stat(Sim,varargin)
%--------------------------------------------------------------------------
% sim_stat function                                                ImBasic
% Description: Given a set of images, calculate statistics of each image.
% Input  : - Images for which to calculate statistics.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) Cell array of matrices.
%            (5) A file contains a list of image (e.g., '@list').
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'CCDSEC' - CCD section or CCD section header keywords
%                       in which to calculate statistics.
%                       If empty use entire image. Default is empty.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure array of statistical properties of images.
%            Statistic includes:
%            .Min, .Max, .Mean, .Median, .StD,
%            .NAXIS1, .NAXIS2, (self explanatory)
%            and:
%            .NumNAN     - Number of NaN pixels in image.
%            .Percentile - 1,2,3 sigma lower and upper percentiles.
%                          see err_cl.m for details.
%            .MeanErr    - Estimated error in the mean,
%                          calculated using the 68-percentile divide
%                          by sqrt of number of pixels.
%            .Mode       - Mode of the data calculated
%                          by looking for the most frquent data
%                          in a binned histogram in which the
%                          bin size is set by the MeanErr.
%          - Structure array of all the images (SIM).
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Stat=sim_stat('*.fits');
% Reliable: 2
%-----------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.CCDSEC = [];

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


if (ischar(InPar.CCDSEC)),
    ImSec = [];
else
    ImSec = InPar.CCDSEC;
    InPar.CCDSEC = [];
end

Sim = images2sim(Sim,varargin{:},'ImSec',ImSec);
Nim = numel(Sim);

Stat = struct('NAXIS1',cell(Nim,1),...
              'NAXIS2',cell(Nim,1),...
              'Min',   cell(Nim,1),...
              'Max',   cell(Nim,1),...
              'Mean',  cell(Nim,1),...
              'NumNAN',cell(Nim,1),...
              'Median',cell(Nim,1),...
              'StD',   cell(Nim,1),...
              'Mode',  cell(Nim,1),...
              'Percentile',cell(Nim,1),...
              'MeanErr',cell(Nim,1));

for Iim=1:1:Nim,
    if (isempty(InPar.CCDSEC)),
        % use full image
        Image  = Sim(Iim).(ImageField);
    else
        CCDSEC = get_ccdsec_head(Sim(Iim).(HeaderField),InPar.CCDSEC);
        if (any(isnan(CCDSEC))),
            error('Problem with CCDSEC');
        end
    
        Image  = Sim(Iim).(ImageField)(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
    end
    
    %--- Calculate Image statistics ---        
    Stat(Iim).NAXIS1 = size(Image,2);     % size of x-dimension
    Stat(Iim).NAXIS2 = size(Image,1);     % size of y-dimension
    Stat(Iim).Min    = minnd(Image);      % Minimum value
    Stat(Iim).Max    = maxnd(Image);      % Maximum value
    Stat(Iim).Mean   = nanmean(Image(:));     % Mean of image
    Stat(Iim).NumNAN = sum(isnan(Image(:))); % Number of NaNs

    Stat(Iim).Median = nanmedian(Image(:));   % Median of image
    Stat(Iim).StD    = nanstd(Image(:));      % StD of image
    % 1,2,3 sigma lower and upper percentiles:
    [Stat(Iim).Mode,Stat(Iim).Percentile,Stat(Iim).MeanErr]=mode_image(Image);

end

        