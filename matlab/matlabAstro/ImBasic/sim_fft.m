function Sim=sim_fft(Sim,varargin)
%--------------------------------------------------------------------------
% sim_fft function                                                 ImBasic
% Description: Calculate FFT or inverse FFT of structure images array
%              (or SIM class).
% Input  : - Set of images to fft or inverse fft.
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
%            'Op'      - FFT or IFFT operation {@fft2|@ifft2}.
%                        Default is @fft2.
%            'Pad'     - FFT padding to size of [Nrows, Ncols].
%                        Default is size(Image).
%            'OverPad' - FFT padding specified in factor relative
%                        to image, rather than the final size.
%                        Default is 1.
%            'Shift'   - FFT shift {true|false}. Default is true.
%            'FFTIm'   - FFT image field (ImageField) {true|false}.
%                        Default is true.
%            'FFTMask' - FFT mask image field (MaskField) {true|false}.
%                        Default is false.
%            'FFTErrIm'- FFT error image field (ErrImField) {true|false}.
%                        Default is false.
%            'FFTBack' - FFT background image field (BackImField) {true|false}.
%                        Default is false.
%            'FFTWeight' - FFT weight image field (WeightImField) {true|false}.
%                        Default is false.
%            'ImageField'- Image field in the SIM structure.
%                        Default is 'Im'.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of FFT or IFFT images.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_fft('lred012*.fits');
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;


ImageField     = 'Im';
HeaderField    = 'Header';
%FileField      = 'ImageFileName';
MaskField      = 'Mask';
BackImField    = 'BackIm';
ErrImField     = 'ErrIm';
WeightImField  = 'WeightIm';

DefV.Op          = @fft2;   
DefV.Pad         = [];
DefV.OverPad     = 1;
DefV.Shift       = true;
DefV.FFTIm       = true;
DefV.FFTMask     = false;
DefV.FFTErrIm    = false;
DefV.FFTBack     = false;
DefV.FFTWeight   = false;
DefV.ImageField  = ImageField;
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

ImageField = InPar.ImageField;


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

for Iim=1:1:Nim,
    if (isempty(InPar.Pad)),
        Pad = size(Sim(Iim).(ImageField)).*InPar.OverPad;
    else
        Pad = InPar.Pad.*InPar.OverPad;
    end
        
    % fft image
    if (isfield(Sim(Iim),ImageField) && InPar.FFTIm),
        Sim(Iim).(ImageField) = InPar.Op(Sim(Iim).(ImageField),Pad(1),Pad(2));
        if (InPar.Shift)
            Sim(Iim).(ImageField) = fftshift(fftshift( Sim(Iim).(ImageField), 1), 2);
        end
    end
    
    % fft mask
    if (isfield(Sim(Iim),MaskField) && InPar.FFTMask),
        Sim(Iim).(MaskField) = InPar.Op(Sim(Iim).(MaskField),Pad(1),Pad(2));
        if (InPar.Shift)
            Sim(Iim).(MaskField) = fftshift(fftshift( Sim(Iim).(MaskField), 1), 2);
        end
    end
    
    % fft error image
    if (isfield(Sim(Iim),ErrImField) && InPar.FFTErrIm),
        Sim(Iim).(ErrImField) = InPar.Op(Sim(Iim).(ErrImField),Pad(1),Pad(2));
        if (InPar.Shift)
            Sim(Iim).(ErrImField) = fftshift(fftshift( Sim(Iim).(ErrImField), 1), 2);
        end
    end
    
    % fft background image
    if (isfield(Sim(Iim),BackImField) && InPar.FFTBack),
        Sim(Iim).(BackImField) = InPar.Op(Sim(Iim).(BackImField),Pad(1),Pad(2));
        if (InPar.Shift)
            Sim(Iim).(BackImField) = fftshift(fftshift( Sim(Iim).(BackImField), 1), 2);
        end
    end
    
    % fft weight image
    if (isfield(Sim(Iim),WeightImField) && InPar.FFTWeight),
        Sim(Iim).(WeightImField) = InPar.Op(Sim(Iim).(WeightImField),Pad(1),Pad(2));
        if (InPar.Shift)
            Sim(Iim).(WeightImField) = fftshift(fftshift( Sim(Iim).(WeightImField), 1), 2);
        end
    end
    
    
    
     %--- Update header ---
     if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
     end
     Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('FFT operation : %s',char(InPar.Op))});
                                    
    
end
