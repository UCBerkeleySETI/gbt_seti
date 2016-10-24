function [PSF,StarsCube,XY]=psf_builder(Sim,varargin)
%--------------------------------------------------------------------------
% psf_builder function                                              ImPhot
% Description: Construct the PSF for an images by averaging selected stars.
% Input  : - Image for which to calculate/build the PSF.
%            This can be a FITS image, a matrix or SIM (single image).
%            Note that the image should be background subtracted (see also
%            the SbBack options).
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'StarList' - List of [X,Y] coordinates of stars to use
%                         for PSF construction.
%                         If empty, will attempt to look for such stars.
%                         Default is empty.
%            'SelectPSF'- Function handle of function to use in order to
%                         select PSF stars.
%                         The program get an image and returne two vectors
%                         [St]=FUN(Image,AddPar). Where AddPar are
%                         additional parameters (see 'SelectPSFpar').
%                         St should be a structure array with a field named
%                         .XY containing the [X,Y] coordinates of the
%                         selected PSF stars.
%                         Default is @select_psf_stars
%            'SelectPSFpar'- Cell array of additional parameters to
%                         pass to the SelectPSF function.
%                         Default is {}.
%            'BoxHalfSize'- PSF box half size. Default is 10.
%                         The box size (even or odd) depands on
%                         'CenterOnPixEdge'.
%            'CenterOnPixEdge' - If true then the box size will be an odd
%                         by odd number of pixels (=BoxHalfSize.*2+1)
%                         and the PSF will be centered on 0.0. (pixel
%                         edge). If false then box size will be even by
%                         even number of pixels (=BoxHalfSize.*2) and the
%                         PSF will be centered on 0.5 (pixel center).
%                         Default is true.
%            'Sampling' - Pixel sampling. Default is 1.
%            'MaxStars' - Maximum number of stars to use for PSF
%                         construction. Default is 100;
%            'InterpMethod'- Interpolation method. See interp2.m for
%                         options. Default is 'cubic'.
%            'RadPhot'  - Radius of aperture photometry for flux
%                         normalization. Default is 5.
%                         This radius will be used to perform aperture
%                         photometry on each PSF star and to normalize them
%                         to the same level.
%            'RadMoment' - Radius of aperture for moments calculation.
%                         Default is 7.
%            'BackMethod' - Background subtraction method. If 'none' will
%                         not subtract background, otherwise will call
%                         sim_background. Default is 'mode_fit'.
%            'FunCombPSF' - Function handle to use for combining all
%                         PSFs. Default is @nanmedian.
%            'FunStdPSF' - Function handle to use for std estimation of
%                         PSFs. Default is @nanstd.
%            'FunErrPSF' - Function handle to use for error estimation of
%                         PSFs. Default is @mean_error.
%            'NormPSF' - Normalize PSF integral to this value.
%                        If empty then do not normalize.
%                        Default is 1.
%            'ZeroEdgesThresh' - A threshold in S/N units. 
%                        If ZeroEdgesMethod='pix', then pixels in the
%                        PSF which have S/N lower than this threshold will
%                        be set to zero.
%                        If ZeroEdgesMethod='rad' then all the pixel outside
%                        a radius of  sqrt(total number of pixels / pi)
%                        will be set to zero. If empty, then do not apply.
%                        Default is 3.
%            'ZeroEdgesMethod' - {'pix'|'rad'}. Default is 'rad'.
%                        See 'ZeroEdgesThresh' parameter for details.
%            'Rad0'    - radius in PSF stamp outside to make the PSF equal
%                        to zero. Default is BoxHalfSize.
%            'OrigScale' - A flag indicating if to retun image to its
%                        original scale {true|false}. Default is true.
%            'ReSizeMethod' - Method to use for returning the PSF to its
%                        original scale. See imresize.m for options.
%                        Default is 'lanczos2'.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: image2sim.m, sim_background.m
% Output : - A structure containing the PSF. The following fields are
%            available:
%            'PSF'
%            'ErrPSF'
%            'VecX'
%            'VecY'
%          - Cube of all stars from which the PSF is constructed.
%            The third dimension corresponds to the star number.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: PSF=psf_builder(Im,'BackMethod','mode_fit','StarList',SL)
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.StarList       = [];
DefV.SelectPSF      = @psf_select; %@select_psf_stars;
DefV.SelectPSFpar   = {};
DefV.BoxHalfSize    = 10;
DefV.CenterOnPixEdge= true; %true;
DefV.Sampling       = 1; %4
DefV.MaxStars       = 100;
DefV.InterpMethod   = 'cubic'; %'linear';
DefV.RadPhot        = 2; %4; %2;   % 5    % radius of aperture to use for aper phot flux normalization
DefV.RadMoment      = 5;       % radius of aperture to use for moments calculation
DefV.BackMethod     = 'mode_fit'; %none';
DefV.FunCombPSF     = @nanmedian;
%DefV.FunStdPSF      = @nanstd;
%DefV.FunErrPSF      = @mean_error;
DefV.NormPSF        = 1;
DefV.RejectMethod   = 'minmax';
DefV.Reject         = [2 2];
DefV.ZeroEdgesThresh= 2;
DefV.ZeroEdgesMethod= 'rad';
DefV.Rad0           = 6;
DefV.OrigScale      = true;
DefV.ReSizeMethod   = 'lanczos2';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.Rad0)),
    InPar.Rad0 = min(InPar.BoxHalfSize);
end

% set RadPhot and RadMoment to be smaller than half box size
InPar.RadPhot   = min(InPar.RadPhot,min(InPar.BoxHalfSize));
InPar.RadMoment = min(InPar.RadMoment,min(InPar.BoxHalfSize));


% Read single image
Sim = images2sim(Sim,varargin{:},'ReadHead',false);
Nim = numel(Sim);


if (isempty(InPar.StarList)),
    SelectedPSFstars = InPar.SelectPSF(Sim,InPar.SelectPSFpar{:});
else
    SelectedPSFstars.XY = InPar.StarList;
end
Nimpsf = numel(SelectedPSFstars);


% subtract background
switch lower(InPar.BackMethod)
    case 'none'
        % do nothing
    otherwise
        %Sim = sim_background(Sim,varargin{:},'SubBack',true);
        [Sim,BS] = sim_back_std(Sim,varargin{:},'SubBack',true);
end


if (InPar.CenterOnPixEdge),
    Edge   = 0;
    Center = 0.0;
else
    Edge   = 0;   % used to be 1?!
    Center = 0.5;
end

% grid of PSF
VecX    = (-InPar.BoxHalfSize:1:InPar.BoxHalfSize-Edge).';
VecY    = VecX;
VecXpsf = (-InPar.BoxHalfSize:1./InPar.Sampling:InPar.BoxHalfSize-Edge).';
VecYpsf = VecXpsf;
[MatXpsf,MatYpsf] = meshgrid(VecXpsf,VecYpsf);
Npix    = numel(VecXpsf);  % number of pixels on a side of the PSF stamp
    
    
%PSF = simdef(Nim,1);
PSF = struct_def({'Nstar','PSF','StdPSF','NpixUse','ErrPSF','VecX','VecY','X1','Y1','X2','Y2','XY','SigmaX','SigmaY','Rho','FWHMs','FWHMa'},Nim,1);
for Iim=1:1:Nim,
    % for each image
    XYpsf = SelectedPSFstars(min(Nimpsf,Iim)).XY;
    XY(Iim).XY = XYpsf;
    Nstars = size(XYpsf,1);

    ImSize = size(Sim(Iim).(ImageField));
    VecXim = (1:1:ImSize(2));
    VecYim = (1:1:ImSize(1));

    StarsCube = zeros(Npix,Npix,Nstars);              % init Stars cube
    StarsFlux = zeros(Nstars,1);
    FlagInAper = (MatXpsf.^2 + MatYpsf.^2)<InPar.RadPhot.^2;
    
    % go over all the stars and interpolate their light into a grid:
    if (Nstars==0),
        PSF(Iim).Nstar  = 0;
        PSF(Iim).SigmaX = NaN;
        PSF(Iim).SigmaY = NaN;
        PSF(Iim).Rho    = NaN;
        PSF(Iim).FWHMs  = NaN;
        PSF(Iim).FWHMa  = NaN;
    else
        for Istars=1:1:Nstars,    
            % interpolate star into StarsCube
            ImageCut = interp2fast(VecXim,VecYim,...
                                   Sim(Iim).(ImageField),...
                                   MatXpsf+XYpsf(Istars,1)+Center,...
                                   MatYpsf+XYpsf(Istars,2)+Center,...
                                   InPar.InterpMethod, NaN);
            %StarsFlux(Istars) = nansum(ImageCut(FlagInAper));
            StarsFlux(Istars) = sum(ImageCut(FlagInAper));
            StarsCube(:,:,Istars) = ImageCut; %./StarsFlux(Istars);

        end
        
        % remove stars with NaN flux
        FlagOK = ~isnan(StarsFlux);
        StarsFlux = StarsFlux(FlagOK);
        StarsCube = StarsCube(:,:,FlagOK);

        Nstars = numel(StarsFlux);
        ScaleS = zeros(Nstars,1);
        %IndRef = 1;
        [~,IndRef] = max(sum(sum(StarsCube,1),2));
        Ref    = StarsCube(:,:,IndRef);
        for Istars=1:1:Nstars,
            Tmp = StarsCube(:,:,Istars);
            
            ScaleS(Istars) = median(Tmp(FlagInAper)./Ref(FlagInAper));
            StarsCube(:,:,Istars) = StarsCube(:,:,Istars)./ScaleS(Istars);

        end

        % select brightest stars
        if (Nstars>InPar.MaxStars),
            [~,SI] = sort(StarsFlux);
            Ind    = SI(end-InPar.MaxStars+1:end);
            StarsFlux = StarsFlux(Ind);
            StarsCube = StarsCube(:,:,Ind);
        end
        Nstars = numel(StarsFlux);
        PSF(Iim).Nstar = Nstars;

    %     % least square solution of PSF
    %     P = numel(MatXpsf);
    %     Q = Nstars;
    %     H = sparse([],[],[],P.*Q,P+Q,2.*P.*Q);
    %     Y = zeros(P.*Q,1);
    %     IndS = (1:1:P).';
    %     for Iq=1:1:Q,
    %         H(IndS+P.*(Iq-1),Iq) = 1;
    %         H(IndS+P.*(Iq-1),Q+1:Q+P) = diag(ones(1,P));
    %         Tmp = StarsCube(:,:,Istars);
    %         Tmp(Tmp<=0) = NaN;            
    %         Y(IndS+P.*(Iq-1)) = log10(Tmp);
    %     end
    %     Flag = ~isnan(Y);
    %     Par = H(Flag,:)\Y(Flag);
    %     PSF(Iim).PSF1 = 10.^reshape(Par(Q+1:end),size(MatXpsf));
    %     

        % normalize Stars flux
        %StarsCube  = bsxfun(@times,StarsCube,1./StarsFlux);


        % mean PSF and Store PSF
        [PSF(Iim).PSF,PSF(Iim).StdPSF,PSF(Iim).NpixUse] = clip_image_mean(StarsCube,'MeanFun',InPar.FunCombPSF,'RejectMethod',InPar.RejectMethod,'Reject',InPar.Reject);

        PSF(Iim).PSF    = squeeze(InPar.FunCombPSF(StarsCube,3));

        % std and error
        %PSF.StdPSF = squeeze(InPar.FunStdPSF(StarsCube,0,3)).*Norm;
        %PSF.ErrPSF = squeeze(InPar.FunErrPSF(StarsCube,3)).*Norm;
        PSF(Iim).ErrPSF = PSF(Iim).StdPSF./sqrt(PSF(Iim).NpixUse);
        PSF(Iim).VecX   = VecX;
        PSF(Iim).VecY   = VecY;

        % smooth edges
        if (~isempty(InPar.ZeroEdgesThresh)),
            switch lower(InPar.ZeroEdgesMethod)
                case 'pix'
                    PSF(Iim).PSF(PSF(Iim).PSF./PSF(Iim).ErrPSF < InPar.ZeroEdgesThresh) = 0;
                case 'rad'
                    NaboveThresh = sum(PSF(Iim).PSF(:)./PSF(Iim).ErrPSF(:) > InPar.ZeroEdgesThresh);
                    RadSN = sqrt(NaboveThresh./pi);
                    PSF(Iim).PSF((MatXpsf(:).^2 + MatYpsf(:).^2)>RadSN.^2) = 0;
                otherwise
                    error('Unknwon ZeroEdgesMethod option');
            end
        end
        % smooth outside Rad0
        PSF(Iim).PSF((MatXpsf(:).^2 + MatYpsf(:).^2)>InPar.Rad0.^2) = 0;


        % calculate moments
        FlagInAper = (MatXpsf.^2 + MatYpsf.^2)<InPar.RadMoment.^2;

        PSF(Iim).X1     = sum(PSF(Iim).PSF(FlagInAper).*MatXpsf(FlagInAper))./sum(PSF(Iim).PSF(FlagInAper));
        PSF(Iim).Y1     = sum(PSF(Iim).PSF(FlagInAper).*MatYpsf(FlagInAper))./sum(PSF(Iim).PSF(FlagInAper));

        PSF(Iim).X2     = sum(PSF(Iim).PSF(FlagInAper).*MatXpsf(FlagInAper).^2)./sum(PSF(Iim).PSF(FlagInAper)) - PSF(Iim).X1.^2;
        PSF(Iim).Y2     = sum(PSF(Iim).PSF(FlagInAper).*MatYpsf(FlagInAper).^2)./sum(PSF(Iim).PSF(FlagInAper)) - PSF(Iim).Y1.^2;
        PSF(Iim).XY     = sum(PSF(Iim).PSF(FlagInAper).*MatXpsf(FlagInAper).*MatYpsf(FlagInAper))./sum(PSF(Iim).PSF(FlagInAper)) - PSF(Iim).X1.*PSF(Iim).Y1;


        % fit Gaussian
        [Beta]=fit_gauss2d(MatXpsf,MatYpsf,PSF(Iim).PSF,ones(size(PSF(Iim).PSF)),[],false);
        %[Normalization, X0, Y0, SigmaX, SigmaY, Rho, Background].
        PSF(Iim).SigmaX = Beta(4);
        PSF(Iim).SigmaY = Beta(5);
        PSF(Iim).Rho    = Beta(6);

        % FWHM estimated from the geometric mean of SigmaX and SigmaY
        PSF(Iim).FWHMs  = sqrt(PSF(Iim).SigmaX.*PSF(Iim).SigmaY).*2.35;

        % FWHM estimated from the area that contains pixels higher than half the
        % maximum
        PSF(Iim).FWHMa  = 2.*sqrt(numel(find(PSF(Iim).PSF(:)./max(PSF(Iim).PSF(:))>0.5))./pi)./InPar.Sampling;

        % change sampling to output sampling
        % TBD
        if (InPar.OrigScale && InPar.Sampling~=1),
            PSF(Iim).PSF = imresize(PSF(Iim).PSF,1./InPar.Sampling,InPar.ReSizeMethod);
            PSF(Iim).StdPSF = imresize(PSF(Iim).StdPSF,1./InPar.Sampling,'triangle');
            PSF(Iim).NpixUse = imresize(PSF(Iim).NpixUse,1./InPar.Sampling,'triangle');
            PSF(Iim).ErrPSF = imresize(PSF(Iim).ErrPSF,1./InPar.Sampling,'triangle');
        end

        % normalization
        SumPSF = sum(PSF(Iim).PSF(:));
        PSF(Iim).SumPSF = SumPSF;
        if (isempty(InPar.NormPSF)),
            Norm = 1;
        else
            %SumPSF = sum(PSF(Iim).PSF(:));
            Norm   = InPar.NormPSF./SumPSF;
        end
        PSF(Iim).PSF    = PSF(Iim).PSF.*Norm;
        PSF(Iim).StdPSF = PSF(Iim).StdPSF.*Norm;
        PSF(Iim).ErrPSF = PSF(Iim).ErrPSF.*Norm;
        % make sure 0 pixels are set to zero in Std and Err
        PSF(Iim).StdPSF = PSF(Iim).StdPSF.* double(PSF(Iim).PSF>0);
        PSF(Iim).ErrPSF = PSF(Iim).ErrPSF.* double(PSF(Iim).PSF>0);

        PSF(Iim).Sum    = sum(PSF(Iim).PSF(:));
    end
end





