function OutImageName=sim2fits(Sim,varargin)
%--------------------------------------------------------------------------
% sim2fits function                                                ImBasic
% Description: Write a single elemnet of a structure image array (SIM)
%              as a FITS file. See sims2fits.m for writing multiple files.
% Input  : - A single elemnet of a structure image array (SIM).
%            See image2sim.m for definitions.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'OutName'   - Output file name. Default is 'Image.fits'.
%            'Prefix'    - A string prefix to add before the output file
%                          name. Default is ''.
%            'Suffix'    - A string suffix to add after the output file
%                          name. Default is ''.
%            'OutDir'    - A directory name in which to write the FITS
%                          images to. Default is ''.
%                          If NaN then will attempt to take the file name
%                          from the SIM FileField field.
%            'TmpName'   - Write the output as an image with temporary
%                          image name in the /tmp directory {true|false}.
%                          Default is false. If true, will override the
%                          OutName option.
%            'WriteHead' - Write image header {true|false}.
%                          Default is true.
%            'ImageField'- A string containing the image field name in
%                          the SIM. Default is 'Im'.
%            'DataType'  - See fitswrite_my.m for options. Default is
%                         'single'.
% Output : - Output image name.
% See also: sims2fits.m, image2sim.m, images2sim.m
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Aug 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutImageName=sim2fits(Sim);
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.OutName    = 'Image.fits';
DefV.Prefix     = '';
DefV.Suffix     = '';
DefV.OutDir     = '';
DefV.TmpName    = false;
DefV.WriteHead  = true;
DefV.ImageField = ImageField;
DefV.DataType   = 'single';
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (numel(Sim)>1),
    error('sim2fits.m works on a single image - use sims2fits.m instead');
end

if (InPar.TmpName),
    InPar.OutName = tempname;
end

% If OutName is NaN then attempt to read file name from FileField
if (isnan(InPar.OutName)),
    if (isfield(Sim, FileField)),
        if (~isempty(Sim.(FileField))),
            InPar.OutName = Sim.(FileField);
        else
            error('No image name in SIM');
        end
    else
        error('No image name in SIM');
    end
end
            

OutImageName = sprintf('%s%s%s%s',InPar.OutDir,InPar.Prefix,InPar.OutName,InPar.Suffix);


if (InPar.WriteHead)
    if (isfield(Sim,HeaderField)),
        Header = Sim.(HeaderField);
    else
        Header = [];
    end
else    
    Header = [];
end
fitswrite_my(Sim.(InPar.ImageField),sprintf('%s%s',InPar.OutDir,OutImageName),Header,InPar.DataType);

    