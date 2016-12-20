function Sim=sim_crdetect(Sim,varargin)
%--------------------------------------------------------------------------
% sim_crdetect function                                            ImBasic
% Description: For images stored in a structure array (SIM) -
%              Find and remove cosmic rays in an astronomical image
%              using the L.A.cosmic algorithm (van Dokkum 2001).
% Input  : - Image or a cell array of images or a structure array
%            of images.
%          * Arbitrary number of pairs of input arguments ...,key,val,...
%            The following keywords are available:
%            'CleanCR'- Clean image from CRs (true|false). If false then
%                     the original image will be stored, but the mask
%                     will be updated with the CR positions.
%                     Default is true.
%            'Gain' - CCD gain for noise estimation. Default is 1.
%            'RN'   - CCD read noise [e-] for noise estimation.
%                      Default is 10.
%            'Nsigma' - CR detection threshold in sigma. Default is 10.
%            'Fth'    - Fine structure threshold. Default is 2.
%                       If FWHM is provided than this parameter is
%                       overrided.
%            'FWHM'   - PSF FWHM to estimate 'Fth' based on Figure 4
%                       in van Dokkum (2001).
%            'Mask'   - Bit mask in which to set a bit for each pixel
%                       which is affected by CR. This mask will be shared
%                       among all the images in the input structure array.
%                       This parameter is overided by the "Mask" field
%                       in the structure array.
%            'Bit_CR' - Index of bit in the bit mask to flag.
%                       Either an index or a function handle
%                       that get 'Bit_CR' and returns the Bit_CR index.
%            'MaskType' - Type of bit mask to create. Default is 'uint16'.
%            'IntMethod'- inpaint_nans.m interpolation method. Default is 2.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, imcrdetect.m, iminterp.m
% Output : - A structure array with the cleaned images and the mask images.
% Reference: http://www.astro.yale.edu/dokkum/lacosmic/
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_crdetect(Im);
% Reliable: 2
%--------------------------------------------------------------------------
FunName = mfilename;


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

% parameters
DefV.CleanCR  = true;
DefV.Gain     = 1;
DefV.RN       = 10;
DefV.Nsigma   = 8;
DefV.Fth      = 2;
DefV.FWHM     = []; %2.5;  % pix
DefV.Mask     = [];
DefV.Bit_CR   = [];
DefV.MaskType = 'uint16';
DefV.IntMethod = 2;
DefV.CopyHead     = 'y';
DefV.AddHead      = [];

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim);
Nim = numel(Sim);

% bit index
InPar.Bit_CR = get_bitmask_def(InPar.Bit_CR,'Bit_CR');

for Iim=1:1:Nim,
    if (isfield(Sim(Iim),MaskField)),
        % use Sim(Iim).(MaskField) as is
    else
        if (isempty(InPar.Mask)),
            % no mask
            Sim(Iim).(MaskField) = zeros(size(Sim(Iim).(ImageField)),InPar.MaskType);
        else
            % Mask is provided throuh InPar.Mask
            Sim(Iim).(MaskField) = InPar.Mask;
        end
    end
    if (InPar.CleanCR),
        [Sim(Iim).(MaskField),CleanImage]=imcrdetect(Sim(Iim).(ImageField),varargin{:},'Mask',Sim(Iim).(MaskField));
        Sim(Iim).(ImageField) = CleanImage;
    else
        Sim(Iim).(MaskField)=imcrdetect(Sim(Iim).(ImageField),varargin{:},'Mask',Sim(Iim).(MaskField));
    end
    
    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('CRdetect Nsigma : %f',InPar.Nsigma)});
              
    
end

