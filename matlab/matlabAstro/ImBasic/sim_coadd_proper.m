function [Coadd,PSF]=sim_coadd_proper(Sim,varargin)
%--------------------------------------------------------------------------
% sim_coadd_proper function                                        ImBasic
% Description: Image caoddition, with optional offset, scaling, weighting
%              and filtering.
% Input  : - Input images can be FITS, SIM or other types of
%            images. For input options see images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Align'  - {true|false}. Default is false.
%                       If true then will align the images prior to
%                       coaddition using sim_align_shift.m.
%            'ScaleMethod' - Scale flux method.
%                       See sim_scaleflux.m for details.
%                       options are:
%                       'Const' - Scaling (divide the image ) by a user
%                                 defined constant. Default.
%                       (Value in 'ScaleConst').
%                       Additionl options include:
%                       'mode_fit', 'none', 'median', 'mean', 'std', 'var'
%                       Note that the scaling is done prior to background
%                       subtraction.
%            'ScaleConst' - Scale flux constant. Default is 1.
%                       Since the program require the images to have
%                       gain of 1, you can use this to convert the images
%                       gain to 1.
%            'BackMethod' - Background subtraction method.
%                       See sim_back_std.m for options. 
%                       options include:
%                       'none' - no background subtraction.
%                       'mode_fit' - Fit a Gaussian to the pixels
%                                histogram. Default.
%            'BackMethod'- Background subtraction method.
%                       See sim_back_std.m for options. 
%                       Default is 'mode_fit'.
%            'Sigma'    - Sigma (std) of background noise of the images.
%                       This is either a vector of sigma values (element
%                       per image), or a scalar to be used for all
%                       images, or a string. String options are:
%                       'mode_fit' - Use mode_fit.m to estimate sigma.
%                                    Default.
%                       'ErrIm'    - Read sigma from SIM 'ErrIm' field. 
%            'Transp'   - Images transparency factors
%                       (i.e., these are the effective photometry
%                       zero points in flux units).
%                       This is either a vector of transp values (element
%                       per image), or a scalar to be used for all
%                       images, or empty.
%                       If empty then will use weights4coadd.m to
%                       estimate the transparency.
%                       Default is empty.
%            'Filter' - Filters to cross-correlate each image prior to
%                       coaddition.
%                       This can be a cell array of matrices,
%                       a single matrix, a singel cell or a SIM
%                       array of filters.
%                       Filter size must be an odd numeber.
%                       If empty then use weights4coadd.m to estimate
%                       PSF of each image.
%            'FiltPrep'- Filter prepartion:
%                       'none' - Cross correlate the filter as is.
%                       'shift' - Shift the center of the filter kernel
%                                to the 1,1 corner. The kernel center
%                                is assumed to be in the filter center.
%                        Default is 'shift'.
%            'FiltNorm'- Normalize filters such their sum equal to this
%                        number. Default is 1. If empty, do not normalize.
%                        Default is 1.
%            'Sharp'   - Return the image (true) or the matched filtered
%                        image (false). Default is tue.
%            'Verbose' - {true|false}. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, sim_background.m, sim_scaleflux.m,
%            sim_filter.m, sim2cube.m
% Output : - Coadded image in SIM format.
%          - Effective PSF of coadded image.
% License: This program can be used only for non-profit educational
%          and scientific use.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Coadd,PSF]=sim_coadd_proper('PTF*.fits','Align',true);
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';
WeightImField = 'WeightIm';



DefV.Align              = false;
DefV.ScaleMethod        = 'const';   % {'none'|'mode_fit'|...}
DefV.ScaleConst         = 1;
DefV.BackMethod         = 'mode_fit';   % {'none'|'mode_fit'|...}
DefV.Sigma              = 'mode_fit';   % 'mode_fit|'errim' | vector
DefV.Transp             = [];
%DefV.WeightMethod       = 'const';
%DefV.WeightConst        = 1;
DefV.Filter             = {};
DefV.FiltPrep           = 'shift';
DefV.FiltNorm           = 1;
% DefV.CombMethod         = 'sum';
% DefV.Prctile            = 0.0;
% DefV.Quantile           = 0.5;
% DefV.CombMethodWeight   = 'fracnotnan';
% DefV.CombMethodErr      = 'rss';   
% DefV.CombMethodBack     = 'sum';
DefV.Sharp              = true;
DefV.Verbose            = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim   = images2sim(Sim,varargin{:});
Nim   = numel(Sim);


% align images
if (InPar.Align),
    if (InPar.Verbose),
        fprintf('Align all images\n');
    end
    [Sim]=sim_align_shift(Sim,[],varargin{:});
end



% Scale images
switch lower(InPar.ScaleMethod)
    case 'none'
        % do nothing
    otherwise
        [Sim] = sim_scaleflux(Sim,varargin{:},'ScaleMethod',InPar.ScaleMethod,'ScaleConst',InPar.ScaleConst);
end


% remove background
switch lower(InPar.BackMethod)
    case 'none'
        % do nothing
    otherwise
        if (InPar.Verbose),
            fprintf('Subtract background from all images\n');
        end
        %Sim   = sim_background(Sim,varargin{:},'BackMethod',InPar.BackMethod);
        [Sim]   = sim_back_std(Sim,varargin{:},'BackStdAlgo',InPar.BackMethod);
        if (ischar(InPar.Sigma)),
            switch lower(InPar.Sigma)
                case 'mode_fit'
                    InPar.Sigma = 'errim';
                otherwise
                    % do nothing
            end
        end
end


% get Sigma
if (ischar(InPar.Sigma)),
    switch lower(InPar.Sigma)
        case 'mode_fit'            
            % attempt to estimate Sigma based on image background std
            InPar.Sigma = zeros(Nim,1);
            for Iim=1:1:Nim,
                [~,InPar.Sigma(Iim)] = mode_fit(Sim(Iim).(ImageField));
            end
        case 'errim'
            % read from ErrIm field
            for Iim=1:1:Nim,
                InPar.Sigma(Iim) = nanmean(Sim(Iim).(ErrImField)(:));
            end
            %InPar.Sigma = [Sim.(ErrImField)]';
        otherwise
            error('Unknown Sigma option');
    end
else
    if (numel(InPar.Sigma)==1),
        InPar.Sigma = InPar.Sigma.*ones(Nim,1);
    end
end

% get Transp
if (isempty(InPar.Transp)),
    % use weights4coadd.m
    OutT = weights4coadd(Sim,'CalcAbsZP',true,'CalcVar',false,'CalcPSF',false,'CalcPSF',true,'PSFsame',true);
    InPar.Transp = [OutT.AbsTran].';
else
    if (numel(InPar.Transp)==1),
        InPar.Transp = InPar.Transp.*ones(Nim,1);
    end
end

% get PSF
if (isempty(InPar.Filter)),
    % use weights4coadd.m
    if (exist('OutT','var')),
        % weights4coadd.m already run
        InPar.Filter = OutT.PSF;
    else
        OutT = weights4coadd(Sim,'CalcAbsZP',false,'CalcVar',false,'CalcPSF',true);
        InPar.Filter = OutT.PSF;
    end
end


%--------------------
%--- coadd images ---
%--------------------
for Iim=1:1:Nim,
    
    if (Iim==1),
        % pad main array
        [Sy,Sx] = size(Sim(Iim).(ImageField));
        PadY = double(is_evenint(Sy));
        PadX = double(is_evenint(Sx));
    end
    
    Image = Sim(Iim).(ImageField);
    Image = padarray(Image,[PadY PadX],'post');
    SizeImage = size(Image);
    
    % shift filter such that it will not introduce a shift to the image
    SizeFilter = size(InPar.Filter{Iim});
    PadImY = 0.5.*(SizeImage(1)-(SizeFilter(1)-1)-1);
    PadImX = 0.5.*(SizeImage(2)-(SizeFilter(2)-1)-1);
    Filter = padarray(InPar.Filter{Iim},[PadImY PadImX],'both');
    Filter = InPar.FiltNorm.*Filter./sum(Filter(:));
    switch lower(InPar.FiltPrep)
        case 'shift'
            %Filter = ifftshift(ifftshift(Filter,1),2);
        case 'none'
            % do nothing
        otherwise
            error('Unknown FiltPrep option');
    end
    
    if (Iim==1),
        SumNom  = zeros(SizeImage);
        SumDen  = zeros(SizeImage);
    end
    SumNom   = SumNom + InPar.Transp(Iim)./(InPar.Sigma(Iim).^2).*conj(fft2(Filter)).*fft2(Image);
    SumDen   = SumDen + (InPar.Transp(Iim)./InPar.Sigma(Iim)).^2.*abs(fft2(Filter)).^2;
    
end

Coadd = SIM;
if (InPar.Sharp),
    Coadd.(ImageField) = fftshift(fftshift(ifft2(SumNom./sqrt(SumDen)),1),2);
else
    Coadd.(ImageField) = fftshift(fftshift(ifft2(SumNom),1),2);
end
Coadd.(ImageField) = Coadd.(ImageField)(1:end-PadY,1:1:end-PadX);    

% The PSF
PSF.(ImageField) = fftshift(fftshift(ifft2(sqrt(SumDen)),1),2);
PSF.(ImageField) = PSF.(ImageField)(1:end-PadY,1:1:end-PadX);





