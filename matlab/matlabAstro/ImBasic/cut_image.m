function [SubImage,Offset]=cut_image(Image,SectionPar,SectionType)
%------------------------------------------------------------------------
% cut_image function                                             ImBasic
% Description: Cut subsection of an image (matrix) given the subsection
%              coordinates, or center and size of subsection.
% Input  : - 2D array containing image or alternatively
%            a cell array of 2D images.
%            If image is a scalar instead of a 2D array it will 
%            not do anything.
%          - Section parameters, in one of the following
%            formats defined by SectionType.
%          - Section type format - one of the followings:
%            'boundry' - [xmin xmax ymin ymax], default.
%            'center'  - [xcenter ycenter xhalfsize yhalfsize]
% Output : - Sub image
%          - Two column matrix of offset parameters, relating
%            the original coordinate system to the new
%            coordinate systtem.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Jul 2005
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: sub_image.m, sim_trim.m
% Example: [SubImage,Offset]=cut_image(rand(10,10),[1 5 1 5])
% Reliable: 2
%------------------------------------------------------------------------

if (nargin==2),
   SectionType = 'boundry';
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (iscell(Image)),
   ImageCell = Image;
else
   ImageCell{1} = Image;
end

Nim    = length(ImageCell);
Nsec   = size(SectionPar,1);
if (Nsec==1),
   SectionPar = ones(Nim,1)*SectionPar;
   Nsec       = Nim;
end

ImSize = zeros(Nim,2);
for Iim=1:1:Nim,
   ImSize(Iim,:) = size(ImageCell{Iim});
end

%--- Convert section type to boundry section type ---
switch lower(SectionType)
 case 'boundry'
    Section  = SectionPar;

 case 'center'
    Section  = [SectionPar(:,1)-SectionPar(:,3), SectionPar(:,1)+SectionPar(:,3), SectionPar(:,2)-SectionPar(:,4), SectionPar(:,2)+SectionPar(:,4)];
 otherwise
    error('Unknown SectionType Option');
end

% define the section that is contained within image
ContSection = [max([ones(Nsec,1),              Section(:,1)],[],2),...
               min([ones(Nsec,1).*ImSize(:,2), Section(:,2)],[],2),...
               max([ones(Nsec,1),              Section(:,3)],[],2),...
               min([ones(Nsec,1).*ImSize(:,1), Section(:,4)],[],2)];


for Iim=1:1:Nim,
   if (numel(ImageCell{Iim})==1),
      % Image is scalar - do nothing
      CurrSubImage = ImageCell{Iim};
   else
      CurrSubImage = ImageCell{Iim}(ContSection(Iim,3):ContSection(Iim,4),ContSection(Iim,1):ContSection(Iim,2));
   end

   if (iscell(Image)==0),
      SubImage = CurrSubImage;
   else
      SubImage{Iim} = CurrSubImage;
   end
end


Offset = [ContSection(:,1)-1, ContSection(:,3)-1];
