function [Back,BackSub]=image_background(Image,varargin)
%--------------------------------------------------------------------------
% image_background function                                        ImBasic
% Description: Generate a background image and also a background
%              subtracted image.
% Input  : - Image (matrix).
%          * Arbitrary number of pairs of ...,key,val,...
%            The following keywords are available:
%            'FullMatrix' - Allways return full matrix (true) or
%                       if possible return a scalar background (false).
%                       Default is false.
%            'BackMethod' - One of the following background estimation
%                       method:
%                       'none'  - Do nothing.
%                       'mode_fit' - Mode using Gaussian fitting (default).
%                       'mode'  - Mode of the image.
%                       'mean'  - Mean og the image.
%                       'median'- Median of the image.
%                       'medfilt'- 2D median filter of the image.
%                       'ordfilt'- 2D order filter of the image.
%                       'polys' - Polynomial surface of the image.
%                       'const' - A constant/matrix background stored in
%                                 'BackConst'.
%            'BackConst' - Background constant to use if BackMethod=const.
%                       Default is 0.
%            'MedFiltSize' - Median and order filter box size.
%                       Default is [65 65].
%            'Order'  - Fraction of the order filter. Default is 0.3.
%            'PolyX'  - Row vector of the orders of X polynomials in the fit.
%                     For example: [1 3], will add a term Px1.*X + Px2.*X.^3.
%                     Default is [1 2].
%            'PolyY'  - Row vector of the orders of Y polynomials in the fit.
%                     Default is [1 2].
%            'PolyXY'- A two column matix of the cross terms orders.
%                     For example: [1 1; 3 4], will add a term:
%                     Pxy1.*X.*Y + Pxy2.*X.^3.*Y.^4.
%                     Default is [1 1].
%            'MaxIter'- Maximum number of sigma clipping iterations.
%                     Default is 0 (i.e., no sigma clipping).
%            'ClipMethod'- Sigma clipping method (see clip_resid.m for details).
%                     Default is 'StD'.
%            'ClipMean' - Method by which to calculate the "mean" of the sample
%                     in the sigma clipping process
%                     (see clip_resid.m for details). Default is 'median'.
%            'Clip' - Two elements vector containing the lower and upper
%                     values for the sigma clipping. This is [Lower, Upper]
%                     number of sigmas (positive) below/above the mean
%                     (see clip_resid.m for details).
%                     See clip_resid.m for defaults.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Back,BackSub]=image_background(Im);
%          Back = image_background(Im,'BackMethod','medfilt');
% Reliable: 2
%--------------------------------------------------------------------------

DefV.FullMatrix      = false;
DefV.BackMethod      = 'mode_fit';
DefV.BackConst       = 0;
DefV.MedFiltSize     = [65 65];
DefV.Order           = 0.3;
DefV.PolyX           = [1 2];
DefV.PolyY           = [1 2];
DefV.PolyXY          = [1 1];
DefV.MaxIter         = 0;
DefV.ClipMethod      = 'StD';
DefV.ClipMean        = 'median';
DefV.Clip            = [];
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


switch lower(InPar.BackMethod)
    case 'const'
        Back = InPar.BackConst;
    case 'median'
        Back = nanmedian(Image(:));
    case 'mean'
        Back = nanmean(Image(:));
    case 'mode'
        Back = mode_image(Image);
    case 'mode_fit'
        Back = mode_fit(Image);
    case 'medfilt'
        Back = medfilt2(Image,InPar.MedFiltSize);
    case 'ordfilt'
        Order = round(prod(InPar.MedFiltSize).*InPar.Order);
        Back = ordfilt2(Image,Order,ones(InPar.MedFiltSize));
    case 'polys'
        % fit a polynomial surface
        Size = size(Image);
        X = (1:1:Size(2)).';
        Y = (1:1:Size(1)).';
        [MatX,MatY] = meshgrid(X,Y);
        Res = fit_2d_polysurface(MatX(:),MatY(:),Image(:),[],'X',InPar.PolyX,'Y',InPar.PolyY,...
                                           'Y',InPar.PolyXY,'MaxIter',InPar.MaxIter,...
                                           'Method',InPar.ClipMethod,...
                                           'Mean',InPar.ClipMean,...
                                           'Clip',InPar.Clip);
        Back = Res.PredZ;
        % back to image
        Back = reshape(Back,Size(1),Size(2));
    case {'none','no'}
        Back = 0;
    otherwise
        error('Unknown BackMethod option');
end

% subtract background
if (nargout>1),
    BackSub = Image - Back;
end    

% full matrix
if (InPar.FullMatrix),
    if (numel(Back)==1),
        Back = Back.*ones(size(Image));
    end
end
    