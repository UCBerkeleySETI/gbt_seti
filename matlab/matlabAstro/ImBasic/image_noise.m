function Noise=image_noise(Image,varargin)
%--------------------------------------------------------------------------
% image_noise function                                             ImBasic
% Description: Calculate the teoretical noise image of a real image
%              using its gain and readout noise.
% Input  : - Image (matrix).
%          * Arbitrary number of pairs of ...,key,val,...
%            The following keywords are available:
%            'FullMatrix' - Allways return full matrix (true) or
%                       if possible return a scalar background (false).
%                       Default is false.
%            'NoiseMethod' - One of the following background estimation
%                       method:
%                       'poisson' - Estimated poisson noise (default).
%                       'std'  - Mode of the image.
%                       'order'- Use two order filters to estimate
%                                the 50-percentile, and estimate the std.
%            'NoiseFiltSize' - Order filter box size.
%                       Default is [65 65].
%            'Gain'   - CCD Gain. Default is 1.
%            'RN'     - CCD readout noise. Default is 10 e-.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Noise=image_noise(Im);
% Reliable: 2
%--------------------------------------------------------------------------

DefV.FullMatrix      = false;
DefV.NoiseMethod     = 'poisson';     
DefV.NoiseFiltSize   = [65 65];
DefV.Gain            = 1;
DefV.RN              = 10;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


switch lower(InPar.NoiseMethod)
    case 'poisson'
        Noise = sqrt(Image.*InPar.Gain + InPar.RN.^2)./InPar.Gain;
    case 'std'
        Noise = nanstd(Image(:));
    case 'order'
        Order1 = round(prod(InPar.NoiseFiltSize).*0.25);
        Order2 = round(prod(InPar.BoiseFiltSize).*0.75);
        ImO1   = ordfilt2(Image,Order1,ones(InPar.NoiseFiltSize));
        ImO2   = ordfilt2(Image,Order2,ones(InPar.NoiseFiltSize));
        Noise  = (ImO2 - ImO1)./1.34
    otherwise
        error('Unknown NoiseMethod option');
end

% full matrix
if (InPar.FullMatrix),
    if (numel(Noise)==1),
        Noise = Noise.*ones(size(Image));
    end
end
    
