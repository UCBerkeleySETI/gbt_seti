function [OutImageName]=sims2fits(Sim,varargin)
%--------------------------------------------------------------------------
% sims2fits function                                               ImBasic
% Description: Write all the images stored in a structure image array (SIM)
%              as a FITS files. See sim2fits.m for writing a single files.
% Input  : - A single elemnet of a structure image array (SIM).
%            See image2sim.m for definitions.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'OutName'   - Cell array of output file name.
%                          Default is 'Image_%06d.fits'.
%                          where %06d is the image number.
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
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: sim2fits.m
% Output : - Cell array of output image name.
% See also: sim2fits.m, image2sim.m, images2sim.m
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Aug 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutImageName=sim2fits(Sim);
% Reliable: 2
%--------------------------------------------------------------------------



ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.OutName    = [];
DefV.Prefix     = '';
DefV.Suffix     = '';
DefV.OutDir     = '';
DefV.TmpName    = false;
DefV.WriteHead  = true;
DefV.ImageField = ImageField;
DefV.DataType   = 'single';
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Nim = numel(Sim);

if (isempty(InPar.OutName)),
    if (isnan(InPar.OutName)),
        InPar.OutName = num2cell(nan(Nim,1));
    else
        
        InPar.OutName = cell(Nim,1);
        for Iim=1:1:Nim,
            InPar.OutName{Iim} = sprintf('Image_%06d.fits',Iim);
        end
    end
else
    if (~iscell(InPar.OutName)),
        InPar.OutName = {InPar.OutName};
    end
end

VArg = struct2varargin(InPar);
OutImageName = cell(Nim,1);
for Iim=1:1:Nim,
    OutImageName{Iim} = sim2fits(Sim(Iim),varargin{:},VArg{:},'OutName',InPar.OutName{Iim});
end
