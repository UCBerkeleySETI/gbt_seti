function [FlatCorSim,FlatSim]=sim_flat(Sim,varargin)
%--------------------------------------------------------------------------
% sim_flat function                                                ImBasic
% Description: Given a set of images, construct a flat image (or use
%              a user supplied flat image), correct all the images by
%              the flat image and save it to a structure
%              array of flat field corrected images.
% Input  : - Bias subtracted images which to correct for flat field,
%            and if need from which to construct the flat image.
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
%            'FlatImage' - Flat image. Either a structure array or a
%                          FITS image name containing the flat image
%                          to use.
%                          Default is empty. If empty then will
%                          attempt to create the flat image.
%            'FlatList'  - Set of flat images to use in order to
%                          construct the master flat image.
%                          The valid input types are those excepted by
%                          flat_construct.m.
%                          Default is empty. If empty then will
%                          attempt to look for good flat images to use
%                          using is_flat_image.m
%            'FlatFile'  - If a new flat image is created then this is the
%                          name of the flat image to save.
%                          Default is 'Flat.fits'. If empty, then do
%                          not save a bias image.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            sim_imarith.m, is_flat_image.m, images2sim.m, flat_construct.m
% Output : - Structure array of flat corrected images.
%          - Structure array of the flat image.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: FlatCorSim=sim_flat(BiasSubSim);
% Reliable: 2
%-----------------------------------------------------------------------------



ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.FlatImage       = [];
DefV.FlatList        = [];
DefV.FlatFile        = 'Flat.fits';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

%--- read images ---
Sim = images2sim(Sim,varargin{:});

if (isempty(InPar.FlatImage)),
    % Flat image is not provided
    if (isempty(InPar.FlatList)),
        % list of flat images is not provided
        [IsFlat,IsNotSaturated]=is_flat_image(Sim,varargin{:});
        IsFlat  = IsFlat & IsNotSaturated;
        
        % construct flat image
        FlatSim = flat_construct(Sim(IsFlat),varargin{:});
    else
        % Flat list is provided
        % construct flat image
        FlatSim = flat_construct(InPar.FlatList,varargin{:});
    end
else
    % Flat image is provided
    FlatSim = image2sim(InPar.FlatImage);
end

%--- Subtract bias image from all the images ---
AddHead = {'COMMENT','','Flat subtracted images';...
           'COMMENT','','Created by flat_bias.m written by Eran Ofek'};
FlatCorSim = sim_imarith('In1',Sim,'In2',FlatSim,'Op','./',varargin{:},'AddHead',AddHead);  % including mask and error propgation








