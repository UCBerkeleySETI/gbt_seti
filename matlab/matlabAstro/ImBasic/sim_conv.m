function [Sim]=sim_conv(Sim,varargin)
%--------------------------------------------------------------------------
% sim_conv function                                                ImBasic
% Description: Convolve a set of a structure images with a kernel.
% Input  : - Set of images to to convolve.
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
%            'Kernel' - Convolution kerenl. This can be (i) a matrix;
%                     or (ii) a cell array in which the first cell is an
%                     handle to a function that return the kernel, the
%                     second cell is the semi-width of the kerenel, and
%                     the other (optional) elements are arguments of the
%                     function.
%                     The functions should be called like
%                     Z=fun(X,Y,additional_par);
%                     Default is {@gauss_2d, 10,[3 3],0,[0 0]};
%                     A list of built in functions:
%                     (1) @gauss_2d(X,Y,[SigmaX,SigmaY],Rho,[X0,Y0], MaxRadius,Norm)
%                     by default SigmaY=SigmaX, Rho=0, X0=0, Y0=0.
%                     MaxRadius is an optional parameter that set the kernel
%                     to zero outside the specified radius. Default is Inf.
%                     Norm is a volume normalization constant of the final
%                     matrix. If NaN then donot normalize. Default is 1.
%                     (2) @triangle_2d(X,Y,Base,[X0, Y0],Norm)
%                     A conic (triangle) convolution kernel.
%                     Base - its semi width
%                     [X0, Y0] - center, default is [0 0].
%                     Norm - Volume normalization, default is 1.
%                     (3) @box_2d(X,Y,SemiWidth,[X0 Y0],Norm)
%                     A box (square) convolution kernel.
%                     SemiWidth - The box semi width. If two elements are
%                     specified then these are the semi width
%                     in the X and Y direction, respectively.
%                     [X0, Y0] - center, default is [0 0].
%                     Norm - Volume normalization, default is 1.
%                     (4) @circ_2d(X,Y,Radius,[X0 Y0],Norm)
%                     A circle (cylinder) convolution kernel.
%                     Radius - The circle radius.
%                     [X0, Y0] - center, default is [0 0].
%                     Norm - Volume normalization, default is 1.
%                     (5) @lanczos_2d(X,Y,A,Stretch,[X0,Y0],Norm)
%                     A lanczos convolution kernel.
%                     A   - Order. Default is 2.
%                     Stretch - Stretch factor. Default is 1.
%                     [X0, Y0] - center, default is [0 0].
%                     Norm - Volume normalization, default is 1.
%            'ConvAlgo' - Convolution algorithm:
%                     @conv2 - using conv2.m
%                     @conv_fft2 - using conv_fft2.m (default).
%                     use conv2 when kernel is small and fft2 when
%                     kernel is large.
%            'Size' - A string specifying the size of the output image.
%                     This can be one of the following strings:
%                     'same' - returns the central part of the convolution
%                              that is the same size as the input (default).
%                     'valid'- returns only those parts of the correlation
%                              that are computed without the zero-padded edges.
%                     'full' - returns the full convolution.
%            'ImageField' - The field name in the structure image (Sim)
%                     containing the image to convolve. Default is 'Im'.
%            'ConvMask' - Convolve mask image {true|false}, with a
%                     a kernel a step-version kernel (Kernel>0 = 1).
%                     Default is false.
%            'MaskField' - The field name in the structure image (Sim)
%                     containing the mask image to convolve.
%                     Default is 'Mask'.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                     Default is 'y'.
%            'AddHead' - Cell array with 3 columns containing additional
%                     keywords to be add to the header.
%                     See cell_fitshead_addkey.m for header structure
%                     information. Default is empty matrix.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of convolved images.
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sim]=sim_conv('lred0126.fits');
% Reliable: 2
%--------------------------------------------------------------------------
FunName = mfilename;


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.Kernel      = {@gauss_2d, 10,[3 3],0,[0 0]};
DefV.ConvAlgo    = @conv_fft2;
DefV.Size        = 'same';
DefV.ImageField  = ImageField;
DefV.ConvMask    = false;
DefV.MaskField   = MaskField;
DefV.CopyHead    = 'y';
DefV.AddHead     = {};
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField = InPar.ImageField;
MaskField  = InPar.MaskField;

%--- prepare the Kernel ---
if (isnumeric(InPar.Kernel)),
   ConvKernel = InPar.Kernel;
else
   if (iscell(InPar.Kernel)),
      KernelFun   = InPar.Kernel{1};
      KernelWidth = InPar.Kernel{2};
      KernelPar   = InPar.Kernel(3:end);  %,InPar.AddPar{:}};
      [MatX,MatY] = meshgrid((-KernelWidth:1:KernelWidth),(-KernelWidth:1:KernelWidth));
      ConvKernel  = feval(KernelFun,MatX,MatY,KernelPar{:});
   else
     error('Unknown Kernel type');
   end
end
KernelSize = size(ConvKernel);

%--- prepare the kernel for the mask image ---
if (InPar.ConvMask),
    MaskConvKernel = zeros(size(ConvKernel));
    MaskConvKernel(ConvKernel>0) = 1;
end


%--- read images ---
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

%--- Go over images and convolve ---
for Iim=1:1:Nim,
    %--- convolve image ---  
    %Sim(Iim).(ImageField) = conv2(Sim(Iim).(ImageField),ConvKernel,InPar.Size);
    Sim(Iim).(ImageField) = InPar.ConvAlgo(Sim(Iim).(ImageField),ConvKernel,InPar.Size);

    if (InPar.ConvMask),
        Sim(Iim).(MaskField) = InPar.ConvAlgo(Sim(Iim).(MaskField),MaskConvKernel,InPar.Size);
    end
   
    
     %--- Update header ---
     if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
     end
     Sim(Iim) = sim_update_head(Sim(Iim),'CopyHead',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('Size Method: %s',InPar.Size),...
                                                   sprintf('Convolution kernel size: %d,%d',KernelSize([2 1]))});
                                    
end

    
