function Sim=sim_background(Sim,varargin)
%--------------------------------------------------------------------------
% image_background function                                        ImBasic
% Description: Generate a background image and also a background
%              subtracted image.
% Input  : - Image (matrix).
%          * Arbitrary number of pairs of ...,key,val,...
%            The following keywords are available:
%            'ImageField' - Image field name in the SIM structure array.
%                       Default is 'Im'.
%            'BackImField' - Background field name in the SIM structure
%                       array. Default is 'BackIm'.
%            'StoreBack' - Store background image in SIM {true|false}.
%                       Default is true.
%            'SubBack' - Subtract background from image and store in SIM
%                       {true|false}. Default is true.
%            'FullMatrix' - Allways return full matrix (true) or
%                       if possible return a scalar background (false).
%                       Default is false.
%            'BackMethod' - One of the following background estimation
%                       method:
%                       'none'  - Do nothing.
%                       'mode_fit' - Mode using Gaussian fitting (default).
%                       'mode'  - Mode of the image.
%                       'mean'  - Mean of the image.
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
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
% Output : - SIM with background image an optionaly background subtracted.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_background('lred012[5-6].fits','BackMethod','Polys');
%          Sim=sim_background('lred012[5-6].fits','BackMethod','medfilt');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.ImageField      = ImageField;
DefV.BackImField     = BackImField;
DefV.StoreBack       = true;
DefV.SubBack         = true;
DefV.OutSIM          = true;

% image_background.m parameters
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField  = InPar.ImageField;
BackImField = InPar.BackImField;

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end


for Iim=1:1:Nim,
    [Back,BackSub]=image_background(Sim(Iim).(ImageField),varargin{:});
    if (InPar.StoreBack),
        Sim(Iim).(BackImField) = Back;
    end
    if (InPar.SubBack),
        Sim(Iim).(ImageField) = BackSub;
    end
end