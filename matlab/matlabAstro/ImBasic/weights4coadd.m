function [Out,W]=weights4coadd(Sim,varargin)
%--------------------------------------------------------------------------
% weights4coadd function                                           ImBasic
% Description: Calculate image weights for optimal coaddition under
%              various assumptions. Also return imaages statistics
%              including the images Std, Variance, Background, RN, Gain,
%              ZP, relative Transperancy, PSF, SigmaX, SigmaY, Rho and
%              FWHM.
% Input  : - List of images. See images2sim.m for options.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'CalcVar'   - A flag indicating if to estimate the images
%                          variance. Default is true.
%            'CalcBack'  - A flag indicating if to estimate the images
%                          background. Default is false.
%            'CalcRN'    - A flag indicating if to estimate the images
%                          Read noise. Default is false.
%            'CalcGain'  - A flag indicating if to estimate the images
%                          gain. Default is false.
%            'CalcRelZP' - A flag indicating if to estimate the images
%                          relative zero point using relative photometry
%                          (see phot_relzp.m). Default is false.
%            'CalcAbsZP' - A flag indicating if to estimate the images
%                          absolute zero point. Default is true.
%                          This is done by constructing the PSF (using
%                          the same stars) for all the images and
%                          comparing the PSF fluxes.
%            'CalcPSF'   - A flag indicating if to estimate the images
%                          PSF (see psf_builder.m). Default is true.
%            'PSFsame'   - If true than PSF will be calculated using the
%                          same stars in all the images.
%                          If false than than in each image different
%                          PSF stars may be used. Default is true.
%            'RN'        - Images read noise. This is either a scalar
%                          or vector of images readnoise, or a string
%                          containing the readnoise keyword name in
%                          the header. Default is 'READNOI'.
%            'Gain'      - Images Gain. This is either a scalar
%                          or vector of images gain, or a string
%                          containing the gain keyword name in
%                          the header. Default is 'GAIN'.
%            'MethodStD' - Method by which to estimate the images
%                          background noise. See sim_std.m for options.
%                          Default is 'fit'.
%            'RegisterImages' - Is an image regsiteration is required
%                          {false|true}. Default is false (i.e., assuming
%                          images are already registered). This is needed
%                          for the PSF and zero point measurements.
%            'MagColName' - Catalog column name containing the source
%                          magnitudes. Default is 'MAG_AUTO'.
%            'MagErrColName' - Catalog column name containing the source
%                          magnitude errors. Default is 'MAGERR_AUTO'.
%            'MaxMeanErr' - Maximum mean error of sources to use in
%                          the relative photometry solution.
%                          Default is 0.05 mag.
%            'MinMeanErr' - Minimum mean error of sources to use in
%                          the relative photometry solution.
%                          Default is 0.003 mag.
%            'Verbose'  - Verbose {true|false}. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            image2sim.m, images2sim.m, sim_std.m, sim_background.m,
%            sim_getkeyval.m, sim_align_shift.m, simcat_matchcoo.m,
%            addcat2sim.m, phot_relzp.m, psf_builder.m
% Output : - A structure containing the following fields:
%            .Std - Vector of std of the images.
%            .Var - Vector of variance of the images.
%            .Back - Background of images.
%            .RN  - Read noise of images.
%            .Gain - Gain of images.
%            .ZP  - Relative zero point of images (magnitude).
%            .Tran - Relative transperancy of images.
%            .PSF - A cell array of PSFs of the images.
%            .SigmaX - Best fit Gaussian Sigma in X direction.
%            .SigmaY - Best fit Gaussian Sigma in Y direction.
%            .Rho - Best fit Gaussian Rho.
%            .FWHM - FWHM of PSFs.
%          - A structure containing various weights for images. The
%            following fields are available:
%            .EV - E/V (relative transperancy divided by variance).
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Out,W]=weights4coadd('PTF_201210*.resamp.fits');
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';
%WeightImField = 'WeightIm';


DefV.CalcVar           = true;
DefV.CalcBack          = false;
DefV.BackStdBlockN     = [4 4];
DefV.CalcRN            = false;
DefV.CalcGain          = false;
DefV.CalcRelZP         = false;
DefV.CalcAbsZP         = true;
DefV.CalcPSF           = true;      % including sigma
DefV.PSFsame           = true;
DefV.RN                = 'READNOI'; % either images readnoise or header keyword
DefV.Gain              = 'GAIN'; % either images readnoise or header keyword
DefV.MethodStD         = 'fit';
DefV.RegisterImages    = false;  
DefV.MagColName        = 'MAG_AUTO';
DefV.MagErrColName     = 'MAGERR_AUTO';
DefV.MaxMeanErr        = 0.05;
DefV.MinMeanErr        = 0.003;
DefV.Verbose           = true;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

% for each image
Out = struct_def({'Std','Var','Back','RN','Gain','ZP','Tran','PSF','SigmaX','SigmaY','Rho','FWHM'},1,1);

% estimate variance of images background
if (InPar.CalcVar || InPar.CalcBack),
    if (InPar.Verbose),
        fprintf('Estimate variance of images background\n');
    end
    
    [~,BackStd] = sim_back_std(Sim,varargin{:},'BlockN',InPar.BackStdBlockN');
    Out.Std = cell2mat(cellfun(@meannd,{BackStd.StD},'UniformOutput',false)).';
    Out.Var = Out.Std.^2;
    Out.Back = cell2mat(cellfun(@meannd,{BackStd.Back},'UniformOutput',false)).';
    %Sim = sim_std(Sim,varargin{:},'MethodStD',InPar.MethodStD);
    %[Out.Std] = deal(Sim.(ErrImField));
    %Out.Std = [Sim.(ErrImField)].';
    %Out.Var = Out.Std.^2;
end

% % estimate background of images
% if (InPar.CalcBack),
%     if (InPar.Verbose),
%         fprintf('Estimate background\n');
%     end
%     Sim = sim_background(Sim,varargin{:},'SubBack',false);
%     %[Out.Back] = deal(Sim.(BackImField));
%     Out.Back= [Sim.(BackImField)].';
% end

% get RN from image headers
if (InPar.CalcRN),
    if (InPar.Verbose),
        fprintf('Get images readnoise\n');
    end
    if (isnumeric(InPar.RN)),
        Out.RN   = InPar.RN.*ones(Nim,1);
    else
        Out.RN   = cell2mat(sim_getkeyval(Sim,InPar.RN));
    end
end

% get Gain from image headers
if (InPar.CalcGain),
    if (InPar.Verbose),
        fprintf('Get images gain\n');
    end
    if (isnumeric(InPar.Gain)),
        Out.Gain   = InPar.Gain.*ones(Nim,1);
    else
        Out.Gain   = cell2mat(sim_getkeyval(Sim,InPar.Gain));
    end
end

% calculate PSF and Sigma for each image
if (InPar.CalcPSF && ~InPar.PSFsame),
    if (InPar.Verbose),
        fprintf('Estimate images PSF\n');
    end
    if (InPar.RegisterImages),
        [Sim] = sim_align_shift(Sim,[],varargin{:});
        InPar.RegisterImages = false;  % images are already registered
    end
    % add cat to Sim
    Sim = addcat2sim(Sim,varargin{:});
    
    % Number of stars found in each image
    Tmp = cell2mat(cellfun(@size,{Sim.Cat},'UniformOutput',false));
    Out.NstarCat = Tmp(1:2:end).';
    
    %[SelectedPSF]=select_psf_stars(Sim,varargin{:});
    PSF = psf_builder(Sim,varargin{:});
    Out.PSF    = {PSF.PSF};
    Out.Nstar  = [PSF.Nstar].';
    Out.SigmaX = [PSF.SigmaX].';
    Out.SigmaY = [PSF.SigmaY].';
    Out.Rho    = [PSF.Rho].';
    Out.FWHM   = [PSF.FWHMa].'; 
else
    if (InPar.CalcPSF),
        InPar.CalcAbsZP = true;
    end
end


% calculate relative photometric zero point for images
if (InPar.CalcRelZP),
    if (InPar.Verbose),
        fprintf('Estimate images relative photometric zero point\n');
    end
    if (InPar.RegisterImages),
        [Sim] = sim_align_shift(Sim,[],varargin{:});
        InPar.RegisterImages = false;  % images are already registered
    end
    % add cat to Sim
    Sim = addcat2sim(Sim,varargin{:});
    % match sources
    [Mat] = simcat_matchcoo(Sim,[],varargin{:});
    I = find(~isnan(mean(Mat.(InPar.MagColName))) & ...
             mean(Mat.(InPar.MagErrColName),1)<InPar.MaxMeanErr & ...
             mean(Mat.(InPar.MagErrColName),1)>InPar.MinMeanErr);
     
    Stat=phot_relzp(Mat.(InPar.MagColName)(:,I),Mat.(InPar.MagErrColName)(:,I),varargin{:});
    
    %Tmp = num2cell(Stat.ZP);
    %[Out(1:end).ZP] = deal(Tmp{:});
    
    Out.ZP   = -Stat.ZP;
    Out.Tran = 10.^(0.4.*Out.ZP);
    
    %semilogy(nanmedian(Stat.Mag),nanstd(Stat.Mag),'.')

end

% calculate absolute photometric zero point for images
if (InPar.CalcAbsZP),
    if (InPar.Verbose),
        fprintf('Estimate images absolute photometric zero point\n');
    end
    if (InPar.RegisterImages),
        [Sim] = sim_align_shift(Sim,[],varargin{:});
        InPar.RegisterImages = false;  % images are already registered
    end
    
    % select PSF stars in one image for all the other images
    %[SelectedPSF]=select_psf_stars(Sim(1));
    [SelectedPSF]=psf_select(Sim(1));
    PSF = psf_builder(Sim,'StarList',SelectedPSF.XY,varargin{:},'NormPSF',[]); %,'BoxHalfSize',11,'Rad0',11,'RadPhot',4);
    Out.AbsTran = [PSF.Sum].';
    
    if (InPar.PSFsame && InPar.CalcPSF),
        % use the same PSF stars in all the images
        Out.PSF    = {PSF.PSF};
        Out.Nstar  = [PSF.Nstar].';
        Out.SigmaX = [PSF.SigmaX].';
        Out.SigmaY = [PSF.SigmaY].';
        Out.Rho    = [PSF.Rho].';
        Out.FWHM   = [PSF.FWHMa].'; 
    end
    %error('Absolute calibration not yet available');
end

    



if (nargout>1),
    % calculate weights
    W.EV    = Out.AbsTran./Out.Var;
end 