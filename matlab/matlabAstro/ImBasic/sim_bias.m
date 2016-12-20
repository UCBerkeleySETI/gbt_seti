function [BiasSubSim,BiasSim]=sim_bias(Sim,varargin)
%--------------------------------------------------------------------------
% sim_bias function                                                ImBasic
% Description: Given a set of images, construct a bias image (or use
%              a user supplied bias image), subtract the bias image
%              from all the science images and save it to a structure
%              array of bias subtracted images.
% Input  : - Images from which to subtract the bias image
%            and if need from which to construct the bias image.
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
%            'BiasImage' - Bias image. Either a structure array or a
%                          FITS image name containing the bias image
%                          to use.
%                          Default is empty. If empty then will
%                          attempt to create the bias image.
%            'BiasList'  - Set of bias images to use in order to
%                          construct the master bias image.
%                          The valid input types are those excepted by
%                          bias_construct.m.
%                          Default is empty. If empty then will
%                          attempt to look for good bias images to use
%                          using is_bias_image.m
%            'BiasFile'  - If a new bias image is created then this is the
%                          name of the bias image to save.
%                          Default is 'Bias.fits'. If empty, then do
%                          not save a bias image.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            sim_imarith.m, is_bias_image.m, images2sim.m, bias_construct.m
%          - Output : - Structure array of bias subtracted images.
%                     - Structure array of the bias image.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: BiasSubSim=sim_bias('blue*.fits');
% Reliable: 2
%-----------------------------------------------------------------------------

ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.BiasImage       = [];
DefV.BiasList        = [];
DefV.BiasFile        = 'Bias.fits';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

%--- read images ---
Sim = images2sim(Sim,varargin{:});

if (isempty(InPar.BiasImage)),
    % Bias image is not provided
    if (isempty(InPar.BiasList)),
        % list of bias images is not provided
        [IsBias,IsGoodNoise,IsGoodMean]=is_bias_image(Sim,varargin{:});
        IsBias  = IsBias & IsGoodNoise & IsGoodMean;
        
        % construct bias image
        BiasSim = bias_construct(Sim(IsBias),varargin{:});
    else
        % Bias list is provided
        % construct bias image
        BiasSim = bias_construct(InPar.BiasList,varargin{:});
    end
else
    % Bias image is provided
    BiasSim = image2sim(InPar.BiasImage);
end

%--- Subtract bias image from all the images ---
AddHead = {'COMMENT','','Bias subtracted images';...
           'COMMENT','','Created by sim_bias.m written by Eran Ofek'};
BiasSubSim = sim_imarith('In1',Sim,'In2',BiasSim,'Op','-',varargin{:},'AddHead',AddHead);  % including mask and error propgation








