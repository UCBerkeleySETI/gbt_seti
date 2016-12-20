function [Object,AllMag,AllMagErr,AllFlux,AllFluxErr,AllArea,ObjInfo]=aperphot(Mat,Coo,varargin)
%------------------------------------------------------------------------------
% aperphot function                                                     ImPhot
% Description: Aperture photometry. Given a matrix and a list of coordinates,
%              calculate accurate centers for each object and perform
%              aperture photometry.
% Input  : - Matrix or a FITS file name.
%          - List of coordinates [X, Y] (i.e., [J, I]).
%          * Arbitrary number of pairs of parameters (keyword, value,...):
%            'VerifyKey'  - Verify keyword option {'y'|'n'}. If 'n',
%                           then ignore errornous keyword options,
%                           default is 'y'.
%                           Note that this keyword must be the first keyword.
%            'MaxIter'    - Maximum number of centering iterations,
%                           (0 for no centering) default is 1.
%            'CenBox'     - Width of centering box, default is 7.
%            'Aper'       - Radius of aperture in [pix] (scalar or vector),
%                           default is 5.
%            'ZP'         - Photometry zero point [mag], default is 22.
%            'Gain'       - gain, default is 1.
%            'ReadNoise'  - Readout noise [e], default is 6.5.
%            'Annu'       - Radius of sky annulus [pix],
%                           default is 12.
%            'DAnnu'      - Width of sky annulus [pix],
%                           default is 10.
%            'SkyAlgo'    - Sky subtraction algorithm:
%                           'Mean'     - mean
%                           'Median'   - median, default.
%                           'PlaneFit' - plane fit.
%            'SkySigClip' - Sigma clipping for sky rejuction,
%                           default is [3.5 3.5] sigmas,
%                           for lower and upper sigma clip.
%                           If NaN, then no sigma clipping. 
%            'Print'      - print results to screen: {'y' | 'n'},
%                           default is 'y'.
%            'OffsetX'    - Offset in X position to add to all
%                           measurments of X position, default is 0.
%            'OffsetY'    - Offset in Y position to add to all
%                           measurments of X positio, default is 0.
% Output : - Structure array of parameters per object.
%          - Magnitude for all stars (row for stars, column for aperture).
%          - Magnitude Err for all stars (row for stars, column for aperture).
%          - Flux for all stars (row for stars, column for aperture).
%          - Flux Err for all stars (row for stars, column for aperture).
%          - Aperure area for all stars (row for stars, column for aperture).
%          - Object information:
%            [M1(1),M1(2),M2(1,1),M2(2,2),M2(1,2),Sky,SkySTD,Nsky];
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Jun 2005
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------

ShiftConv = 1e-5;   % shift convergence criterion


DefV.VerifyKey  = 'y';
DefV.MaxIter    = 1;
DefV.CenBox     = 7;
DefV.Aper       = 5;
DefV.ZP         = 22;
DefV.Gain       = 1.0;
DefV.ReadNoise  = 6.5;
DefV.Annu       = 12;
DefV.DAnnu      = 10;
DefV.SkyAlgo    = 'Median';
DefV.SkySigClip = 3.5;
DefV.Print      = 'y';
DefV.OffsetX    = 0;
DefV.OffsetY    = 0;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


if (ischar(Mat)),
   Mat = fitsread(Mat);
end

if (length(InPar.SkySigClip)==1),
   InPar.SkySigClip = [InPar.SkySigClip, InPar.SkySigClip];
end


Size  = size(Mat);
Nst   = size(Coo,1);    % number of coordinates in list
Naper = length(InPar.Aper); 
% for each star

AllMag      = zeros(Nst,Naper);
AllMagErr   = zeros(Nst,Naper);
AllFlux     = zeros(Nst,Naper);
AllFluxErr  = zeros(Nst,Naper);
AllArea     = zeros(Nst,Naper);
ObjInfo     = zeros(Nst,8);

Object = [];
for Ist=1:1:Nst,
   Object(Ist).MaxVal = NaN;

   %-------------------------------
   %--- Find center and moments ---
   %-------------------------------
   Iter = 0;
   X  = Coo(Ist,1);
   Y  = Coo(Ist,2);
   Shift = ShiftConv.*10;
   while (Shift>ShiftConv && Iter<InPar.MaxIter),
      Iter     = Iter + 1;
      [M1,M2]  = calc_moments1(Mat,X,Y,InPar.CenBox,Size);
      Shift    = plane_dist(M1(1),M1(2),X,Y);
      X        = M1(1);
      Y        = M1(2);
   end

   Object(Ist).X = X;
   Object(Ist).Y = Y;
   
   if (Shift>ShiftConv),
      % center not converged or not attempted to converge
      ConvCenter = 0;
      if (InPar.MaxIter==0),
         %--- calculate moments ---
         [M1,M2] = calc_moments1(Mat,X,Y,InPar.CenBox,Size);
         M1      = Coo(Ist,:);
      end
   else
      % center converged
      ConvCenter = 1;
   end

   %[ConvCenter, M1, M2]

   %---------------------------
   %--- Aperture photometry ---
   %---------------------------
   [MatX,MatY] = meshgrid((1.5:1:Size(2)+0.5),(1.5:1:Size(1)+0.5));
   Dist2       = (MatX - M1(1)).^2 + (MatY - M1(2)).^2;

   
   
   %---------------------
   %--- Calculate sky ---
   %---------------------
   I              = find(Dist2>InPar.Annu.^2 & Dist2<=(InPar.Annu+InPar.DAnnu).^2);
   Nsky           = length(I);

   switch InPar.SkyAlgo
    case 'Median'
       Sky     = median(mat2vec(Mat(I)));
       SkySTD  = std(mat2vec(Mat(I)));

       if (isnan(InPar.SkySigClip)==0),
          J = find(Mat(I)>=(Sky-InPar.SkySigClip(1).*SkySTD) & Mat(I)<=(Sky+InPar.SkySigClip(2).*SkySTD));
          Nsky    = length(J);
          Sky     = median(mat2vec(Mat(I(J))));
          SkySTD  = std(mat2vec(Mat(I(J))));
       end

    case 'Mean'
       Sky     = mean(mat2vec(Mat(I)));
       SkySTD  = std(mat2vec(Mat(I)));

       if (isnan(InPar.SkySigClip)==0),
	      J = find(Mat(I)>=(Sky-InPar.SkySigClip(1).*SkySTD) & Mat(I)<=(Sky+InPar.SkySigClip(2).*SkySTD));
          Nsky    = length(J);
          Sky     = mean(mat2vec(Mat(I(J))));
          SkySTD  = std(mat2vec(Mat(I(J))));
       end
    case 'PlaneFit'
       VecX     = mat2vec(MaxX(I));
       VecY     = mat2vec(MaxY(I));
       Val      = mat2vec(Max(I));
       Nsky     = length(I);
       H        = [ones(Nsky,1), VecX, VecY, VecX.*VecY];
       Par      = H\Val;
       Res      = Val - H*Par;
       SkySTD   = std(Res);
       if (isnan(InPar.SkySigClip)==0),
          J = find(Res>=(SkySTD.*InPar.SkySigClip(1)) & Res<=(SkySTD.*InPar.SkySigClip(2)));
          VecX     = mat2vec(MaxX(I(J)));
          VecY     = mat2vec(MaxY(I(J)));
          Val      = mat2vec(Max(I(J)));
          Nsky     = length(J);
          H        = [ones(Nsky,1), VecX, VecY, VecX.*VecY];
          Par      = H\Val;
          Res      = Val - H*Par;
          SkySTD   = std(Res);
          Sky      = mean(Val);   %<- use mean instead of plane...
                                  % assuming source is symetric...
       end

    otherwise
       error('Unknown SkyAlgo Option');
   end


   %---------------------------
   %--- Calculate magnitude ---
   %---------------------------
   
   % maximum
   Object(Ist).MaxVal = max(Mat(I(:)));
   
   Flux        = zeros(1,Naper);
   FluxErr     = zeros(1,Naper);
   Mag         = zeros(1,Naper);
   MagErr      = zeros(1,Naper);
   Area        = zeros(1,Naper);
   for Iaper=1:1:Naper,
      I              = find(Dist2<=InPar.Aper(Iaper).^2);
      Area(Iaper)    = length(I);
      Flux(Iaper)    = sum(sum(Mat(I))) - Sky.*Area(Iaper);
      FluxErr(Iaper) = sqrt(Flux(Iaper)./InPar.Gain + Area(Iaper).*SkySTD.^2 + (Area(Iaper).*SkySTD).^2./Nsky);
      Mag(Iaper)     = InPar.ZP - 2.5.*log10(Flux(Iaper));
      MagErr(Iaper)  = 1.0857.*FluxErr(Iaper)./Flux(Iaper);
   end

   %-------------------------------
   %--- Save Photometry results ---
   %-------------------------------
   AllMag(Ist,:)     = Mag;
   AllMagErr(Ist,:)  = MagErr;
   AllFlux(Ist,:)    = Flux;
   AllFluxErr(Ist,:) = FluxErr;
   AllArea(Ist,:)    = Area;
   ObjInfo(Ist,:)    = [M1(1)+InPar.OffsetX,M1(2)+InPar.OffsetY,M2(1,1),M2(2,2),M2(1,2),Sky,SkySTD,Nsky];

   Object(Ist).Mag      = Mag;
   Object(Ist).MagErr   = MagErr;
   Object(Ist).Flux     = Flux;
   Object(Ist).FluxErr  = FluxErr;
   Object(Ist).AperArea = Area;
   Object(Ist).Sky      = Sky;
   Object(Ist).SkyStD   = SkySTD;
   Object(Ist).SkyArea  = Nsky;
   
   
   switch InPar.Print
    case 'n'
       % do nothing
    case 'y'
       disp(sprintf('\n'))
       disp(sprintf('-------------------------------------------------'))
       disp(sprintf('Star Number = %d ',Ist))
       disp(sprintf('X=%f   Y=%f [Started at: %f %f]',ObjInfo(Ist,[1 2]),Coo(Ist,:) ))
       for Iaper=1:1:Naper,
          disp(sprintf('Aper=%f     Mag = %7.3f +/- %6.3f ',InPar.Aper(Iaper),AllMag(Ist,Iaper),AllMagErr(Ist,Iaper)  ))
       end
       disp(sprintf('Sky=%f    SkySTD=%f    Nsky=%d ',ObjInfo(Ist,[6 7 8]) ))
       disp(sprintf('Ixx=%f   Iyy=%f  Ixy=%f ',ObjInfo(Ist,[3 4 5]) ))

    otherwise
       error('Unknown Print Option');
   end


end





%------------------------------------
%--- Calculate center and moments ---
%------------------------------------
function [M1,M2]=calc_moments1(Mat,X,Y,CenBox,Size)
%------------------------------------
if (nargin==4),
   Size = size(Mat);
end

RangeX   = [round(X-0.5.*CenBox):1:round(X+0.5.*CenBox)].';
RangeY   = [round(Y-0.5.*CenBox):1:round(Y+0.5.*CenBox)].';
Iinrange = find(RangeX>=1 & RangeX<=Size(2));
RangeX   = RangeX(Iinrange);
Iinrange = find(RangeY>=1 & RangeY<=Size(1));
RangeY   = RangeY(Iinrange);
      
SubMat   = Mat(RangeY,RangeX);

[M1,M2]  = moment_2d(SubMat,RangeX,RangeY);   

%---------------------------------
%--- End calc_moments function --
%---------------------------------

