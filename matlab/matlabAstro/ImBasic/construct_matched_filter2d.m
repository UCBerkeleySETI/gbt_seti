function [MF,ShiftedMF]=construct_matched_filter2d(varargin)
%--------------------------------------------------------------------------
% construct_matched_filter2d function                              ImBasic
% Description: Construct an optimal 2D matched filter for sources
%              detection in an image (given by S/(N^2)).
% Input  : * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'MFtype' - One of the following PSF shapes.
%                       Note that if 'PSF' (i.e., nirmerical PSF) is
%                       provided, then this option is ignored.
%                       The following options are available:
%                       'gauss' - A Gaussian. Default.
%                       'circ'  - A flat circle.
%                       'annulus' - A flat annulus.
%                       'circx' - A first moment X-value in a circle.
%                       'circy' - A first moment Y-value in a circle.
%                       'circx2' - A second moment X^2-value in a circle.
%                       'circy2' - A second moment Y^2-value in a circle.
%                       'circxy' - A second moment X*Y-value in a circle.
%                       'circgx' - A first moment X-value weighted by a
%                                  Gaussian.
%                       'circgy' - A first moment Y-value weighted by a
%                                  Gaussian.
%            'ParMF'  - Matched filter parameters.
%                       For each different PSFtype this is interpreted in
%                       a different way. The interpreation for different
%                       'PSFtype' is as follows:
%                       'Gauss' - then these are the Gaussians
%                            parameter described in 'Gauss'.
%                       'circ' - a single parameter with the radius of the
%                            circle.
%                       'annulus' - a flat annulus which inner and outer
%                            radii are [R1, R2].
%                       'circx' | 'circy' | 'circx2' | 'circy2' | 'circxy'-
%                            a single parameter with the radius of the
%                            circle.
%                       'circgx' | 'circgy' - [Gaussian_Sigma, Circ_Radius]
%            'Gauss'  - Parameters of Gaussian PSF. Note that ParMF
%                       override this parameter.
%                       This is an arbitrary number of lines, each
%                       containing 4 columns [sigmaX, sigmaY, cov, Norm]
%                       of a Gaussian. The PSF is constructed from
%                       a combination of all Gaussians.
%                       Default is [3 3 0 676].
%                       The normalization of 676 is selected such that
%                       the S/N for an aperture photometry, with the
%                       other default parameters will be 5.
%                       See optimal_phot_aperture.m
%            'PSF'    - Numerical PSF (override 'Gauss'). Default is [].
%            'FiltHalfSize' - The half size of the filter [x,y]. This is
%                       not ised if PSF is provided. Default is [7 7].
%            'Rad0'   - Zero the matched filter outside this radius.
%                       If empty then will set it to the minimum of
%                       FiltHalfSize.
%                       Default is empty.
%            'Thresh' - Values in the PSF smaller than this number will
%                       be set to zero. Default is empty (do nothing).
%            'B'      - Background [e-]. Default is 100.
%            'RN'     - Readout noise [e-]. Default is 5.
%            'SourceInVar' - Include the source term in the variance (N^2)
%                       or not {true|false}. Default is false.
%            'OnlyPSF'- Return a match filter that contains only the PSF
%                       (i.e., not divided by the noise) {true|false}.
%                       Default is true.
%            'Norm'   - Normalize the matched filter such that the
%                       integral of the matched filter will be equal
%                       to the normalization. Default is 1.
%                       If empty, do not normalize.
% Output : - A matrix of matched filter.
%          - The matched filter shifted using fftshift to the image corner.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Sep 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Res=optimal_phot_aperture('B',150,'Sigma',3);
%          MF=construct_matched_filter2d('B',150,'Gauss',[3 3 0 Res.S]);
%          % construct an MF equal to the PSF
%          MF=construct_matched_filter2d('B',0,'RN',0,'SourceInVar',false,'OnlyPSF',true);
% Reliable: 2
%--------------------------------------------------------------------------

DefV.MFtype         = 'Gauss';
DefV.ParMF          = [];
DefV.Gauss          = [3 3 0 676];   % sigmaX, sigmaY, cov, Norm
DefV.PSF            = [];
DefV.FiltHalfSize   = [7 7];
DefV.AddCol0        = true;
DefV.AddRow0        = true;
DefV.Rad0           = 7;
DefV.Thresh         = [];
DefV.B              = 100;
DefV.RN             = 5;
DefV.SourceInVar    = false;
DefV.OnlyPSF        = true;
DefV.Norm           = 1;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (numel(InPar.FiltHalfSize)==1),
    InPar.FiltHalfSize = [InPar.FiltHalfSize, InPar.FiltHalfSize];
end

if (isempty(InPar.Rad0)),
    InPar.Rad0 = min(InPar.FiltHalfSize);
end

if (~isempty(InPar.PSF)),
    % use InPar.PSF instead of InPar.Sigma
else
    VecX   = (-InPar.FiltHalfSize(1):1:InPar.FiltHalfSize(1)).';
    VecY   = (-InPar.FiltHalfSize(2):1:InPar.FiltHalfSize(2)).';
    [MatX,MatY] = meshgrid(VecX,VecY);
    MatR2       = MatX.^2 + MatY.^2;
    InPar.PSF   = zeros(size(MatX));  % init matched filter
    switch lower(InPar.MFtype)
        case 'gauss'
            % construct PSF from sigma
            if (~isempty(InPar.ParMF)),
                InPar.Gauss = InPar.ParMF;
            end
            Ngauss = size(InPar.Gauss,1);  % number of Gaussian components in PSF        
            for Igauss=1:1:Ngauss,
                Rho     = InPar.Gauss(Igauss,3);
                OneRho2 = sqrt(1 - Rho.^2);
                SigmaX  = InPar.Gauss(Igauss,1);
                SigmaY  = InPar.Gauss(Igauss,2);
                Norm    = InPar.Gauss(Igauss,4);
                Z       = (MatX./SigmaX).^2 + (MatY./SigmaY).^2 + 2.*Rho.*MatX.*MatY./(SigmaX.*SigmaY);

                InPar.PSF = InPar.PSF + Norm./(2.*pi.*SigmaX.*SigmaY.*OneRho2).*exp(-Z./(2.*(1-Rho.^2)));

            end
        case 'circ'
            % construct a flat circle
            InPar.PSF(MatR2<=InPar.ParMF.^2) = 1;
            InPar.Rad0 = InPar.ParMF;
            
        case 'annulus'
            % construct a flat annulus
            InPar.PSF(MatR2>InPar.ParMF(1).^2 & MatR2<=InPar.ParMF(2).^2) = 1;
            InPar.Rad0 = max(InPar.ParMF);
            
        case 'circx'
            % construct a X first moment filter in a circle
            InPar.PSF = MatX;
            InPar.PSF(MatR2>InPar.ParMF.^2) = 0;
            InPar.Rad0 = InPar.ParMF;
            
        case 'circy'
            % construct a Y first moment filter in a circle
            InPar.PSF = MatY;
            InPar.PSF(MatR2>InPar.ParMF.^2) = 0;   
            InPar.Rad0 = InPar.ParMF;
            
        case 'circx2'
            % construct a X^2 second moment filter in a circle
            InPar.PSF = MatX.^2;
            InPar.PSF(MatR2>InPar.ParMF.^2) = 0;    
            InPar.Rad0 = InPar.ParMF;
            
        case 'circy2'
            % construct a Y^2 second moment filter in a circle
            InPar.PSF = MatY.^2;
            InPar.PSF(MatR2>InPar.ParMF.^2) = 0;   
            InPar.Rad0 = InPar.ParMF;
            
        case 'circxy'
            % construct a X*Y second moment filter in a circle
            InPar.PSF = MatX.*MatY;
            InPar.PSF(MatR2>InPar.ParMF.^2) = 0;  
            InPar.Rad0 = InPar.ParMF;
            
        case 'circgx'
            % construct a X first moment filter weighted by a Gaussian
            % pars [sigma, radius]
            InPar.PSF = MatX.*exp(-MatR2./(2.*InPar.ParMF(1).^2));
            InPar.PSF(MatR2>InPar.ParMF(2).^2) = 0;
            InPar.Rad0 = InPar.ParMF(2);
        
        case 'circgy'
            % construct a Y first moment filter weighted by a Gaussian
            % pars [sigma, radius]
            InPar.PSF = MatY.*exp(-MatR2./(2.*InPar.ParMF(1).^2));
            InPar.PSF(MatR2>InPar.ParMF(2).^2) = 0;
            InPar.Rad0 = InPar.ParMF(2);    
        otherwise
            
            error('Unknown PSFtype option');
    end
        
end

if (InPar.OnlyPSF)
    MF = InPar.PSF;
else
    if (InPar.SourceInVar),
        MF = InPar.PSF./(InPar.PSF + InPar.B + InPar.RN.^2);
    else
        MF = InPar.PSF./(InPar.B + InPar.RN.^2);
    end
end

MatR = sqrt(MatX.^2+MatY.^2);
MF(MatR>InPar.Rad0) = 0;
if (~isempty(InPar.Thresh)),
    MF(MF<InPar.Thresh) = 0;
end


if (~isempty(InPar.Norm)),
    MF = MF./sum(MF(:));
    MF = MF.*InPar.Norm;
end

if (nargout>1),
    % prepare ShiftedMF
    ShiftedMF = fftshift(fftshift(MF,1),2);
end

