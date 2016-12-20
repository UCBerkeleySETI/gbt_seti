function FiltIm=filter2smart(Image,Filter)
%--------------------------------------------------------------------------
% filter2smart function                                            ImBasic
% Description: Cross correlate (i.e., filter) a 2D image with a given
%              filter. The function is using either filter2.m or direct
%              fft, according to the filter size compared with the image
%              size.
% Input  : - Images to filter.
%          - A filter.
% Output : - A filtered image which size is equal to the input image size.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Image=zeros(1024,1024); Image(400,400)=1;                
%          Filter=construct_matched_filter2d('FiltHalfSize',128);
%          FiltIm=filter2smart(Image,Filter);
% Reliable: 2
%--------------------------------------------------------------------------



SizeImage  = size(Image);
SizeFilter = size(Filter);
%prod(SizeFilter)./prod(SizeImage)

if (prod(SizeFilter)./prod(SizeImage)>0.2),
    % Filter is comparable in size to Image so use direct fft
    
    if (all(SizeFilter==SizeImage)),
        % do nothing
        PadStatus = false;
    else
        PadStatus = true;
       
        % pad main array
        %[Sy,Sx] = size(Sim(Iim).(ImageField));
        PadY = double(is_evenint(SizeImage(1)));
        PadX = double(is_evenint(SizeImage(2)));
        
        Image = padarray(Image,[PadY PadX],'post');
        SizeImage = size(Image);

        % shift filter such that it will not introduce a shift to the image
        %SizeFilter = size(Filter);
        PadImY = 0.5.*(SizeImage(1)-(SizeFilter(1)-1)-1);
        PadImX = 0.5.*(SizeImage(2)-(SizeFilter(2)-1)-1);
        Filter = padarray(Filter,[PadImY PadImX],'both');
        
    end
    
    % shift filter center to origin
    Filter = ifftshift(ifftshift(Filter,1),2);
    
    % cross-correlation
    FiltIm = ifft2(fft2(Image).*conj(fft2(Filter)));
    %Image = fftshift(fftshift(Image,1),2);
    %Image = ccf_fft2(Image,Filter,false,false);
    if (PadStatus),
        FiltIm = FiltIm(1:end-PadY,1:1:end-PadX);    
    end
    
else
    % Filter is small relative to Image so use filter2.m
    FiltIm=filter2(Filter,Image,'same');
end

    



