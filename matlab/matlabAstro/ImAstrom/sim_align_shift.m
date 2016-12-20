function [AlSim,TranRes]=sim_align_shift(Sim,SimRef,varargin)
%--------------------------------------------------------------------------
% sim_align_shift function                                        ImAstrom
% Description: Given a set of images and a reference image, register
%              (align) the images to the reference image.
%              This function is suitable for transformation which are
%              mostly shifts (with some small rotation and distortions).
% Input  : - A set of images to align. This can be a list of images,
%            FITS images, SIM, and more.
%            See images2sim.m for valid options.
%          - An optional reference image.
%            See images2sim.m for valid options.
%            If empty, or not provided, then will use one of the input
%            images (according to the 'ChooseRef' parameter) as
%            a reference image. Default is [].
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'OutSize' - Size of output images [Y,X].
%                        If empty, size will be identical to reference
%                        image. Default is [].
%            'RePop'   - Repopulate Cat field in SIM (if already exist).
%                        Default is false.
%            'ChooseRef'- If reference image is not provided set one
%                        of the following images as a reference:
%                        'maxn' - image with max. number of sources
%                                 (default).
%                        'first'- first image.
%                        'last' - last image.
%            'ColXc'   - Column name or column index containing the X
%                        coordinates in the catalog to align.
%                        Default is 'XWIN_IMAGE'.
%            'ColYc'   - Column name or column index containing the Y
%                        coordinates in the catalog to align.
%                        Default is 'YWIN_IMAGE'.
%            'ColXr'   - Column name or column index containing the X
%                        coordinates in the reference to align.
%                        Default is 'XWIN_IMAGE'.
%            'ColYr'   - Column name or column index containing the Y
%                        coordinates in the reference to align.
%                        Default is 'YWIN_IMAGE'.
%            'ColErrXc'- Column name or column index containing the X
%                        coordinate errors in the catalog to align.
%                        Default is [].
%            'ColErrYc'- Column name or column index containing the Y
%                        coordinate errors in the catalog to align.
%                        Default is [].
%            'ColErrXr'- Column name or column index containing the X
%                        coordinate errors in the reference to align.
%                        Default is [].
%            'ColErrYr'- Column name or column index containing the Y
%                        coordinate errors in the reference to align.
%                        Default is [].
%            'MatchPars' - A cell array of parameters to pass to 
%                        match_lists_shift.m who is responsible for
%                        matching the catalog against the reference.
%                        Default is {}.
%            'FitTranPars' - A cell array of parameters to pass to 
%                        fit_tran2d.m who is responsible for
%                        fitting the transformation with the matched
%                        sources. Default is {}.
%            'UpdateWCS' - Update WCS by copying it from the reference
%                        image {true|false}. Default is true.
%            'OutSIM' -  Force output to be SIM. Default is true.
%            'Verbose'-  Verbose. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            addcat2sim.m, images2sim.m, image2sim.m
% Output : - SIM structure array containing the aligned (registered)
%            images.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: AlSim=sim_align_shift('PTF*p100037_c02.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';

if (nargin==1),
    SimRef = [];
end

DefV.OutSize        = [];              % [y,x] - like ref image
DefV.RePop          = false;
DefV.ChooseRef      = 'maxn';
DefV.ColXc          = 'XWIN_IMAGE';    % or index
DefV.ColYc          = 'YWIN_IMAGE';
DefV.ColXr          = 'XWIN_IMAGE';
DefV.ColYr          = 'YWIN_IMAGE';
DefV.ColErrXc       = [];    % or index
DefV.ColErrYc       = [];
DefV.ColErrXr       = [];
DefV.ColErrYr       = [];
DefV.MatchPars      = {};
DefV.FunTrans       = @fit_general2d_ns; %@fit_affine2d_ns; %@fit_general2d_ns;
DefV.FunPars        = {'Shift',[],'Rot',[]}; %,...
%                                             'PolyX',[2 1000 1000 0 2000 2000;...
%                                                      0 1000 1000 2 2000 2000;...
%                                                      1 1000 1000 1 2000 2000;...
%                                                      3 1000 1000 0 2000 2000;...
%                                                      0 1000 1000 3 2000 2000],...
%                                             'PolyY',[2 1000 1000 0 2000 2000;...
%                                                      0 1000 1000 2 2000 2000;...
%                                                      1 1000 1000 1 2000 2000;...
%                                                      3 1000 1000 0 2000 2000;...
%                                                      0 1000 1000 3 2000 2000]};


%                                                         
%,'ParAng1',[-1.4294 0 0],...
%                                           'ParAng2',[-1.5009 0 0]};
                                      
%                                            'PolyX',[2 1000 1000 0 2000 2000;...

%                                                     0 2 1000 2000 1000 2000;...
%                                                     1 1 1000 2000 1000 2000;...
%                                                     3 0 1000 2000 1000 2000;...
%                                                     0 3 1000 2000 1000 2000],...
%                                            'PolyY',[2 0 1000 2000 1000 2000;...
%                                                     0 2 1000 2000 1000 2000;...
%                                                     1 1 1000 2000 1000 2000;...
%                                                     3 0 1000 2000 1000 2000;...
%                                                     0 3 1000 2000 1000 2000]};
                                                    
DefV.TransMethod    = 'imtransform';   % 'imwarp'
DefV.UpdateWCS      = true;
DefV.OutSIM         = true;
DefV.Verbose        = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


% read images and catalogs into SIM
Sim    = addcat2sim(Sim,varargin{:},'RePop',InPar.RePop);
Nim    = numel(Sim);
if (InPar.OutSIM),
    AlSim = simdef(Nim,1); %SIM;    % output is of SIM class
end


% Read reference image or use one of the images in SIM
if (isempty(SimRef)),
    switch lower(InPar.ChooseRef)
        case 'first'
            IndRef = 1;
        case 'last'
            IndRef = Nim;
        case 'maxn'
            Nsrc = zeros(Nim,1);
            for Iim=1:1:Nim,
                Nsrc(Iim) = size(Sim(Iim).(CatField),1);
            end
            [~,IndRef] = max(Nsrc);
        otherwise
            error('Unknown ChooseRef option');
    end
    SimRef = Sim(IndRef);
    
    if (InPar.Verbose),
        fprintf('Image number %d will be used as a reference image\n',IndRef);
    end
else 
    SimRef = addcat2sim(SimRef,varargin{:},'RePop',InPar.RePop);
end

if (isempty(InPar.OutSize)),
    InPar.OutSize = size(SimRef.(ImageField));
end

AlSim = simdef(Nim,1);
for Iim=1:1:Nim,
    if (InPar.Verbose),
        fprintf('Match list of image number %d with reference list\n',Iim);
    end
    % get columns
    [~,ColXc,ColYc,ColXr,ColYr,ColErrXc,ColErrYc,ColErrXr,ColErrYr]=col_name2ind(Sim(Iim).(CatColCellField),InPar.ColXc, InPar.ColYc,...
                                                          InPar.ColXr, InPar.ColYr,...
                                                          InPar.ColErrXc, InPar.ColErrYc,...
                                                          InPar.ColErrXr, InPar.ColErrYr);
%     ColXc = Sim(Iim).(CatColField).(InPar.ColXc);
%     ColYc = Sim(Iim).(CatColField).(InPar.ColYc);
%     ColXr = Sim(Iim).(CatColField).(InPar.ColXr);
%     ColYr = Sim(Iim).(CatColField).(InPar.ColYr);
    
    
    % match lists and find intial shift
    [Res,IndBest] = match_lists_shift(Sim(Iim),SimRef,InPar.MatchPars{:},...
                                                      'ColXc',ColXc,...
                                                      'ColYc',ColYc,...
                                                      'ColXr',ColXr,...
                                                      'ColYr',ColYr);
                                                  
    
    % fit transformation                                             
%     [TranRes(Iim)] = fit_tran2d(Sim(Iim).(CatField)(Res(IndBest).IndCat,[ColXc ColYc ColErrXc ColErrYc]), ...
%                                 SimRef.(CatField)(Res(IndBest).IndRef,  [ColXr ColYr ColErrXr ColErrYr]),...
%                                 'FunTrans',InPar.FunTrans,...
%                                 'FunPar',InPar.FunPars);
   
    % fit transformation using the InPar.FunTrans user defined function
%     [TranRes(Iim)] = InPar.FunTrans(Sim(Iim).(CatField)(Res(IndBest).IndCat,[ColXc ColYc ColErrXc ColErrYc]), ...
%                                     SimRef.(CatField)(Res(IndBest).IndRef,  [ColXr ColYr ColErrXr ColErrYr]),...                   
%                                     InPar.FunPars{:});
%                       
    if (isempty(Res)),
        warning('Res is empty - can not find a solution');
    end

    [TranRes(Iim)] = InPar.FunTrans(SimRef.(CatField)(Res(IndBest).IndRef,  [ColXr ColYr ColErrXr ColErrYr]),...
                                    Sim(Iim).(CatField)(Res(IndBest).IndCat,[ColXc ColYc ColErrXc ColErrYc]), ...
                                    InPar.FunPars{:});                


%     [FitRes]=fit_general2d_ns(Sim(Iim).(CatField)(Res(IndBest).IndCat,[ColXc ColYc ColErrXc ColErrYc]), ...
%                               SimRef.(CatField)(Res(IndBest).IndRef,  [ColXr ColYr ColErrXr ColErrYr]),...
%                               {'Shift',[0 0],'Rot',[1 0;0 1],'PolyX',[0 2 2 1000 2000 1000 2000]})
     
    if (InPar.Verbose),
        fprintf('   Number of matches    : %d\n',Res(IndBest).Nmatch);
        fprintf('   Number of stars used : %d\n',TranRes(Iim).Nsrc);
        fprintf('   StdX                 : %f\n',TranRes(Iim).Xrms);
        fprintf('   StdY                 : %f\n',TranRes(Iim).Yrms);
        fprintf('   Std                  : %f\n',TranRes(Iim).rms);
        %fprintf('   Shift X              : %f\n',TranRes(Iim).tdata.ParX(3));
        %fprintf('   Shift Y              : %f\n',TranRes(Iim).tdata.ParY(3));        
    end
    
    % register images
    %AlSim(Iim) = imtransform(Sim(Iim).(ImageField),TranRes.Tran)
    AlSim(Iim) = sim_transform(Sim(Iim),'Trans',TranRes(Iim).Tran,'OutSize',InPar.OutSize,'TransMethod',InPar.TransMethod);
    
    if (InPar.UpdateWCS),
        warning('UpdateWCS option is not implemented yet');
    end
end

                                               
                                               
                                               
                                               