function ImSize=sim_imagesize(Sim,varargin)
%--------------------------------------------------------------------------
% sim_imagesize function                                           ImBasic
% Description: Get the image size of a set of images.
% Input  : - List of images. See images2sim.m for options.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'SizeFromImage' - Get size from image {true|false}.
%                              If false, will attempt to get size from
%                              header. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Two column matrix of image sizes [NAXIS1, NAXIS2].
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ImSize=sim_imagesize(Sim);
%          ImSize=sim_imagesize(Sim,'SizeFromImage',false);
% Reliable: 2
%--------------------------------------------------------------------------



ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.SizeFromImage = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


if (InPar.SizeFromImage),
    Sim = images2sim(Sim,varargin{:},'ReadImage',true);
    Nim = numel(Sim);
    ImSize = zeros(Nim,2);
    for Iim=1:1:Nim,
       ImSize(Iim,:) = fliplr(size(Sim(Iim).(ImageField)));
    end
else
    Sim = images2sim(Sim,varargin{:},'ReadImage',false);
    Nim = numel(Sim);
    ImSize = zeros(Nim,2);
    for Iim=1:1:Nim,
        CellHead = cell_fitshead_getkey(Sim(Iim).(HeaderField),{'NAXIS1','NAXIS2'},'NaN');
        %[KeysVal,KeysComment,Struct]=fits_get_keys(Image,Keys,HDUnum);
        ImSize(Iim,:) = [CellHead{:,2}].';
    end
    
end