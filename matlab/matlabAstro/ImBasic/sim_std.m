function Sim=sim_std(Sim,varargin)
%--------------------------------------------------------------------------
% sim_std function                                                 ImBasic
% Description: Calculate the StD (or error) per image or per pixel for
%              a set of SIM images.
% Input  : - Set of images on which to perfoprm segmentation.
%            Input can be either cell array of images, FITS images names
%            with wild cards, file lists, SIM and more. See images2sim.m
%            for options and help.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'MethodStD' - Method by which to calculate the std:
%                          'poisson' - Poisson statistics (using sqrt(N))
%                                      per pixel and
%                                      taking into account the background
%                                      image from the SimBack parameter
%                                      or the SIM background field, and the
%                                      gain and RN (Bias std image).
%                          'std'     - Standard deviation of image.
%                          'perc'    - Robust StD using 68 percentile.
%                          'fit','mode_fit'- Fit a Gaussian to the image
%                                      histogram using mode_fit.m (default).
%                          Alternatively, this can be a scalar or vector
%                          with a scalar per image to be used as the StD.
%            'SimBack'   - A SIM input or any input that is recognized by
%                          images2sim.m that contains the background
%                          image in the ImageField.
%                          Default is empty. If empty will attempt to read
%                          the background from the BackImField.
%                          If this parameter is empty and if FitBack is
%                          true, then will calculate it using
%                          sim_background.m.
%                          This parameter is used only for
%                          MethodStD='poisson'.
%            'FitBack'   - If SimBack is empty and this parameter is true
%                          than the background will be estimated using
%                          sim_background.m. If false, than will attempt
%                          to use the BackImField. Default is true.
%            'BiasStD'   - A scalar containing the readout noise or an
%                          image of the biase noise (i.e., RN per pixel).
%                          Default is 5.
%            'BiasUnits' - Units of BiasStD {'DN'|'e'}. Default is 'DN'.
%            'AddRN'     - Add the RN in quadrature to the Poisson noise.
%                          Default is false. This is used for 
%                          MethodStD='poisson'.
%            'Gain'      - CCD Gain [e-/DN]. Default is 1.5.
%                          This is used for the image. If BiasUnits='DN'
%                          then this is used also for the BiasStD.
%            'OutSIM'    - Output is a SIM class (true) or a structure
%                          array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: images2sim.m, sim_background.m, mode_fit.m
% Output : - Sim images in which the ErrImField is populated with an
%            error/std image or scalar.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Dec 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: sim_std(rand(1000,1000),'MethodStd','perc')
%          sim_std(5,'BackMethod','mean');
% Reliable: 2
%--------------------------------------------------------------------------
ONE_SIGMA = 0.6827;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';


DefV.MethodStD   = 'fit';        % {'poisson'|'std'|'perc'|'fit' | VALUE}
DefV.SimBack     = [];           % SIM or any other input containing background images
                                 % If empty than fit background using
                                 % sim_background.m
DefV.FitBack     = true;         % if SimBack is empty then states if we
                                 % we need to fit the background or is it
                                 % already available in the SIM background
                                 % field.
DefV.BiasStD     = 5;            % Bias StD or RN
DefV.BiasUnits   = 'DN';         % {'DN'|'e'}
DefV.AddRN       = false;        % Add BiasStd in quadrature...
DefV.Gain        = 1.5;          % Gain [e-/DN]
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

if (~ischar(InPar.MethodStD)),
    StdVal          = InPar.MethodStD;
    InPar.MethodStD = 'value';
    if (numel(StdVal)==1),
        StdVal = StdVal.*ones(Nim,1);
    end
end

switch lower(InPar.MethodStD)
    case 'poisson'  
        if (~isempty(InPar.SimBack)),
            % read SimBack
            SimBack = images2sim(InPar.SimBack,varargin{:});
            Nsb     = numel(SimBack);
        else
            % fit background
            if (InPar.FitBack),
                Sim = sim_background(Sim,varargin{:},'SubBack',false,'StoreBack',true,'ImageField',ImageField);
            end
            Nsb = 0;
        end
    otherwise
        % do nothing
end

switch lower(InPar.BiasUnits),
    case 'dn'
        BiasFactor = InPar.Gain;
    case 'e'
        BiasFactor = 1;
    otherwise
        error('Unknown BiasUnits option');
end

    
for Iim=1:1:Nim,
    switch lower(InPar.MethodStD)
        case 'poisson'            
            if (Nsb==0),
                % use bacground at Sim
                Back = Sim(Iim).(BackImField);
            else
                Back = SimBack(min(Iim,Nsb)).(ImageField);
            end
            
            Sim(Iim).(ErrImField) = sqrt(Back.*InPar.Gain + (double(InPar.AddRN).*InPar.BiasStD.*BiasFactor).^2);
            
        case 'value'
            % StD value supplied by user
            Sim(Iim).(ErrImField) = StdVal(Iim);
        case 'perc'
            Percentile = err_cl(Sim(Iim).(ImageField)(:),ONE_SIGMA);
            Sim(Iim).(ErrImField) = 0.5.*(Percentile(1,2) - Percentile(1,1));
        case {'fit','mode_fit'}
            [~,Sim(Iim).(ErrImField)] = mode_fit(Sim(Iim).(ImageField),varargin{:});
        case 'std'
            Sim(Iim).(ErrImField) = std(Sim(Iim).(ImageField)(:));
        otherwise
            error('Unknown MethodStD option');
    end
    
end
   
               
               
               
               