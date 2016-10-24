function Exam=ds9_exam(Image,varargin)
%--------------------------------------------------------------------------
% ds9_exam function                                                    ds9
% Description: Interactive examination of an image displayed in ds9.
%              The program display an image in ds9 and then prompt
%              the user to use various clicks to examin sources, vectors,
%              and regions.
% Input  : - FITS image file name (string).
%            Alternatively, this can be a matrix or SIM image that will be
%            saved as a temporary FITS file and will be displayed in ds9.
%            If empty matrix than use existing image in the ds9 display.
%            Default is empty.
%          - Frame - This can be a function {'new'|'first'|'last'|'next'|'prev'}
%            or a number, default is empty. If empty use current frame.
%            If the image name is a cell array then
%            the frame number can be a vector, otherwise it will
%            load the images to the sucessive frames.
%            Default is next frame.
%          * Arbitrary number of pairs of ...,key,val,... arguments.
%            The following keywords are available:
%            'CooType'  - Coordinates type in ds9 {'image'}. 
%                         Default is 'image'.
%            'VecStep'  - Interpolation step for "v" option.
%                         Default is 0.3.
%            'InterpMethod' - Interpolation method. Default is 'linear'.
%            'LineLen'  - Line length/box size for the region around
%                         the clicked position.
%            'Radius'   - Radius for centering algorithm, second
%                         moment calculation and PSF fitting.
%                         Default is 10.
%            'Niter'    - Number of centering iterations.
%                         Default is 3.
%            'RadSN'    - Radii at which to calculate S/N in S/N plot.
%                         Default is (0:0.5:15).
%            'RadPSF'   - Radius for PSF fitting. Default is 20.
%            'Figure'   - Index of figure to open. If empty use exsiting
%                         figure. Default is empty.
%            'Plot'     - Plot or type information {true|false}.
%                         Default is true.
%            'PlotPars' - Cell array of parameters to pass to plot lines.
%                         Default is {'k-'}.
%            'SurfPars' - Cell array of parameters to pass to surface.
%                         Default is {}.
%            'ContPars' - Cell array of parameters to pass to contour.
%                         Default is {}.
%            'RadPars'  - Cell array of parameters to pass to radial plot.
%                         Default is {'k.','MarkerSize',8}.
%            'ColX'     - Catalog column name or index in which the X
%                         coordinate is stored. Default is 'XWIN_IMAGE'.
%            'ColY'     - Catalog column name or index in which the X
%                         coordinate is stored. Default is 'YWIN_IMAGE'.
% Output : - A structure aray of the information clicked by the user
%            and returned by each click. Element per click.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% License: GNU general public license version 3
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_disp('20050422051427p.fits',4,'InvertCM','yes')
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
%CatColField     = 'Col';
CatColCellField = 'ColCell';


Mode = 'key';

Def.Image = [];
Def.Frame = [];
if (nargin==0),
    Image = Def.Image;
    Frame = Def.Frame;
elseif (nargin==1),
    Frame = Def.Frame;
else
    % do nothing
end


DefV.CooType        = 'image';
% option: v
DefV.VecStep        = 0.3;
DefV.InterpMethod   = 'linear';
% option: x
DefV.LineLen        = 50;
% option: r
DefV.Radius         = 10;
DefV.Niter          = 3;
% option: n
DefV.RadSN          = (0:0.5:15);
% option: p
DefV.RadPSF         = 20;
% plot
DefV.Figure         = [];
DefV.Plot           = true;
DefV.PlotPars       = {'k-'};
DefV.SurfPars       = {};
DefV.ContPars       = {};
DefV.RadPars        = {'k.','MarkerSize',8};
DefV.ColX           = 'XWIN_IMAGE';
DefV.ColY           = 'YWIN_IMAGE';

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});



ds9_disp(Image,Frame,varargin{:});

IsExit = false;
Ind    = 0;
Exam   = struct('Key',{},'CooX',{},'CooY',{},'Val',{},'VecX',{},'VecY',{},'VecR',{},'VecVal',{});
while (~IsExit),
    Ind = Ind + 1;
    fprintf('Click a key, h for help\n');
    [CooX1,CooY1,Value1,RetKey]=ds9_getcoo(1,InPar.CooType,Mode);
    switch RetKey{1}
        case {'q','Q'}
            IsExit = true;
        case {'h','?'}
            % get help
            ds9_exam_showmenu(InPar);
        case 'E'
            % Edit input arguments
            ParVal = input('Insert parameter to change its value : ','s');
            eval(sprintf('InPar.%s',ParVal));
        case 'v'
            fprintf('Type v on second point\n')
            [CooX2,CooY2,Value2,RetKey] = ds9_getcoo(1,InPar.CooType,Mode);
            switch RetKey{1}
                case {'v','V'}
                    % second v was supplied by user
                    % calculate vector properties
                    Exam(Ind).Key  = RetKey{1};
                    Exam(Ind).CooX = [CooX1, CooX2];
                    Exam(Ind).CooY = [CooY1, CooY2];
                    Exam(Ind).Val  = [Value1, Value2];

                    [Exam(Ind).VecX,Exam(Ind).VecY,Exam(Ind).VecR,Exam(Ind).VecVal,Par]=ds9_getvecprof(Exam(Ind).CooX,Exam(Ind).CooY,varargin{:});
                    
        
                    if (InPar.Plot),
                        if (~isempty(InPar.Figure)),
                            figure(InPar.Figure);
                        end
                        clf;
                        plot(Exam(Ind).VecX,Exam(Ind).VecVal,InPar.PlotPars{:});
                        H = xlabel('X [pix]');
                        set(H,'FontSize',18);
                        H = ylabel('Counts [DN]');
                        set(H,'FontSize',18);
                        %F = inline('polyval(Par,x)','x','Par');
                        F = @(x,Par) polyval(Par,x);
                        multi_axis('x',F,Par);
                        H = xlabel('Y [pix]');
                        set(H,'FontSize',18);
                        drawnow;
                    end
                otherwise
                    fprintf('Second v was not pressed - skip\n')
            end
        case {'x','X'}
            % vector along x axis
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), round(CooY1),...
                              InPar.LineLen,...
                              1];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
                    
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal  = MatVal;
            Exam(Ind).VecR = sqrt( (Exam(Ind).VecX - min(Exam(Ind).VecX)).^2 + ...
                                   (Exam(Ind).VecY - min(Exam(Ind).VecY)).^2);
                    
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 plot(Exam(Ind).VecX,Exam(Ind).VecVal,InPar.PlotPars{:});
                 H = xlabel('X [pix]');
                 set(H,'FontSize',18);
                 H = ylabel('Counts [DN]');
                 set(H,'FontSize',18);
                 drawnow;
             end                   
         case {'y','Y'}
            % vector along y axis
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [round(CooX1), floor(CooY1-InPar.LineLen.*0.5),...
                         1,...
                              InPar.LineLen];
                          
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
                    
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal  = MatVal;
            Exam(Ind).VecR = sqrt( (Exam(Ind).VecX - min(Exam(Ind).VecX)).^2 + ...
                                   (Exam(Ind).VecY - min(Exam(Ind).VecY)).^2);
            
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 plot(Exam(Ind).VecY,Exam(Ind).VecVal,InPar.PlotPars{:});
                 H = xlabel('Y [pix]');
                 set(H,'FontSize',18);
                 H = ylabel('Counts [DN]');
                 set(H,'FontSize',18);
                 drawnow;
            end   
        case {'s','c'}
            % surface
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
                    
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal  = MatVal;
            Exam(Ind).VecR = sqrt( (Exam(Ind).VecX - CooX1).^2 + ...
                                   (Exam(Ind).VecY - CooY1).^2);
                    
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 switch lower(RetKey{1})
                     case 's'
                         surface(Exam(Ind).VecX,Exam(Ind).VecY,Exam(Ind).VecVal,InPar.SurfPars{:});
                         H = zlabel('Counts [DN]');
                         set(H,'FontSize',18);
                         view(30,30);
                         %shading interp
                     case 'c'
                         contour(Exam(Ind).VecX,Exam(Ind).VecY,Exam(Ind).VecVal,InPar.ContPars{:});
                     otherwise
                         error('Unknown RetKey option');
                 end
                 H = xlabel('X [pix]');
                 set(H,'FontSize',18);
                 H = xlabel('Y [pix]');
                 set(H,'FontSize',18);
                 colorbar;
               
                 drawnow;
            end 
        case 'R'
            % radial plot around point (without centering)
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
            MatR = sqrt( (MatX - CooX1).^2 + (MatY - CooY1).^2);
            fprintf('Radial profile manual center X=%f  Y=%f\n',CooX1,CooY1);
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal = MatVal;
            Exam(Ind).VecR = MatR;
                
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 plot(MatR(MatR(:)<InPar.LineLen),MatVal(MatR(:)<InPar.LineLen),InPar.RadPars{:});
                 H = xlabel('Radius [pix]');
                 set(H,'FontSize',18);
                 H = ylabel('Counts [DN]');
                 set(H,'FontSize',18);
                 drawnow;
            end   
          case 'r'
            % radial plot around point (with centering)
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
            Res=moment2d(MatX,MatY,MatVal-median(MatVal(:)),[],[]);
            fprintf('Recenter source to: X=%f  Y=%f\n',Res.X,Res.Y);
            MatR = sqrt( (MatX - Res.X).^2 + (MatY - Res.Y).^2);
            
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal = MatVal;
            Exam(Ind).VecR = MatR;
                
            VecR = (1:1:InPar.LineLen.*0.5);
            Nr   = numel(VecR)-1;
            Exam(Ind).RadProf = zeros(Nr,3);
            for Ir=1:1:numel(VecR)-1,
                Vals = MatVal(MatR>=VecR(Ir) & MatR<VecR(Ir+1));
                Exam(Ind).RadProf(Ir,:) = [(VecR(Ir)+VecR(Ir+1)).*0.5, mean(Vals), std(Vals)./numel(Vals)];
            end
            
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 plot(MatR(MatR(:)<InPar.LineLen),MatVal(MatR(:)<InPar.LineLen),InPar.RadPars{:});
                 H = xlabel('Radius [pix]');
                 set(H,'FontSize',18);
                 H = ylabel('Counts [DN]');
                 set(H,'FontSize',18);
                 drawnow;
            end   
        case {'n','N'}
            % S/N radial plot around a source (with centering)
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
            Res=moment2d(MatX,MatY,MatVal,[],[],'AperR',InPar.RadSN);
            fprintf('Recenter source to: X=%f  Y=%f\n',Res.X,Res.Y);
            %MatR = sqrt( (MatX - Res.X).^2 + (MatY - Res.Y).^2);
            
            Exam(Ind).VecX = [];
            Exam(Ind).VecY = [];
            Exam(Ind).VecVal = Res.SN;
            Exam(Ind).VecR = InPar.RadSN;
                
                  
            if (InPar.Plot),
                if (~isempty(InPar.Figure)),
                     figure(InPar.Figure);
                 end
                 clf;
                 plot(InPar.RadSN,Res.SN,InPar.PlotPars{:});
                 H = xlabel('Radius [pix]');
                 set(H,'FontSize',18);
                 H = ylabel('S/N');
                 set(H,'FontSize',18);
                 drawnow;
            end   
               
         case 'a'
            % aperture photometry (with centering)
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
            Res=moment2d(MatX,MatY,MatVal,[],[],varargin{:});
    
            Exam(Ind).Phot = Res;
            
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal = MatVal;
              
            if (InPar.Plot),
                Res
            end
            
         case {'p','P'}
            % psf photometry (with centering)
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            BoxCoo = [floor(CooX1-InPar.LineLen.*0.5), floor(CooY1-InPar.LineLen.*0.5),...
                              InPar.LineLen,...
                              InPar.LineLen];
            [MatVal,MatX,MatY] = ds9_getbox(BoxCoo,'Corner',InPar.CooType);
            Res=moment2d(MatX,MatY,MatVal,[],[],varargin{:});
            
            MatR = sqrt( (MatX - Res.X).^2 + (MatY - Res.Y).^2);
            FlagR = MatR<InPar.RadPSF;
            Exam(Ind).Phot = Res;
            
            Exam(Ind).VecX = MatX;
            Exam(Ind).VecY = MatY;
            Exam(Ind).VecVal = MatVal;
           
            [Beta,Chi2] = fit_gauss2d(MatX(FlagR),MatY(FlagR),MatVal(FlagR),1,[Res.X,Res.Y]);
            Exam(Ind).ParsPSF = Beta;
            Exam(Ind).Chi2PSF = Chi2;
            Exam(Ind).Chi2Dof = length(find(FlagR));
            
            if (InPar.Plot),
                % Type values of aperture photometry
                Res
                % type PSF fit information
                fprintf('PSF Gaussian fit: Flux=%e X=%f Y=%f\n',Exam(Ind).ParsPSF(1:3));
                fprintf('  SigmaX=%f SigmaY=%f Rho=%f Back=%f\n',Exam(Ind).ParsPSF(4:7));
                fprintf('  chi2/dof=%f/%d\n',Exam(Ind).Chi2PSF,Exam(Ind).Chi2Dof);
            end       
            
        case 'S'
            % search nearest source from Catalog
            Exam(Ind).Key  = RetKey{1};
            Exam(Ind).CooX = CooX1;
            Exam(Ind).CooY = CooY1;
            Exam(Ind).Val  = Value1;
            
            if (issim(Image) || isstruct(Image)),
                if (~isfield_notempty(Image,CatField)),
                    % Cat does not exist
                    % create Cat using addcat2sim
                    Image = addcat2sim(Image);
                end
                
            else
                % not SIM - run SExtractor on image
                Image = addcat2sim(Image);
            end
            % look for nearest source in Cat
            [~,InPar.ColX,InPar.ColY] = col_name2ind(Image.(CatColCellField),InPar.ColX,InPar.ColY);
            D = plane_dist(Image.(CatField)(:,InPar.ColX),Image.(CatField)(:,InPar.ColY),CooX1,CooY1);
            [MinD,MinInd] = min(D);
            
            Exam(Ind).NearestCat = Image.(CatField)(MinInd,:);
            Exam(Ind).ColCell    = Image.(CatColCellField);
            
            Exam(Ind).CatS       = table2struct(array2table(Exam(Ind).NearestCat,'VariableNames',Exam(Ind).ColCell));
            if (InPar.Plot),
                fprintf('Nearest source found: %f pixels from clicked position:\n',MinD);
                Exam(Ind).CatS
            end
            
        otherwise
            fprintf('Uknown key - press h for options\n\n');
    end
end

%-------------------------
function ds9_exam_showmenu(InPar)
%-------------------------

fprintf('--- ds9_exam menu --\n');
fprintf('q   - Quit\n');
fprintf('h/? - Help\n');
fprintf('E   - edit parameters - type "parameter=value" (e.g., "RadPSF=22", "LineLen=30")\n');
fprintf('v   - followed by another v - Vector between two selected points\n');
fprintf('x   - Vector along the x axis (LineLen=%d)\n',InPar.LineLen);
fprintf('y   - Vector along the y axis (LineLen=%d)\n',InPar.LineLen);
fprintf('s   - Surface plot (LineLen=%d)\n',InPar.LineLen);
fprintf('c   - Contour plot (LineLen=%d)\n',InPar.LineLen);
fprintf('r   - Radial plot with centering (LineLen=%d)\n',InPar.LineLen);
fprintf('R   - Radial plot without centering (LineLen=%d)\n',InPar.LineLen);
fprintf('n   - Radial S/N plot with centering (RadSN=%d)\n',max(InPar.RadSN));
fprintf('a   - Aperture photometry with centering (LineLen=%d)\n',InPar.LineLen);
fprintf('p   - PSF Gaussian fit with centering (RadPSF=%d)\n',InPar.RadPSF);
fprintf('S   - Search for source in associated catalog nearest to position\n');




            