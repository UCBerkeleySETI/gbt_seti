function [FitRes,X,Y]=fit_general2d_ns(Cat,Ref,varargin)
%--------------------------------------------------------------------------
% fit_general2d_ns function                                       ImAstrom
% Description: Fit (or apply) a 2D general transformation,
%              non-simultanously to both axes, to a set of control points.
%              The fit is of the form:
%              Xref = A + B*X - C*Y + D*AM*sin(Q1) + E*AM*sin(Q2) + ...
%                     F*AM*Color*sin(Q1) + G*AM*Color*sin(Q1) + ...
%                     PolyX(X) +PolyX(Y) + PolyX(R) + H*FunX(pars)
%              Yref = A + B*X - C*Y + D*cos(Q1) + E*cos(Q2) + ...
%                     F*AM*Color*cos(Q1) + G*AM*Color*cos(Q1) + ...
%                     PolyY(X) +PolyY(Y) + PolyY(R) + H*FunY(pars)
%              If the free parameters are supplied than the Cat input
%              is transformed to the Ref system.
% Input  : - Matrix of the first set of control points (the "catalog")
%            [X, Y, ErrX, ErrY]. If ErrY is not provided, then
%            will assume ErrY=ErrX. If ErrX is not provided then
%            ErrY=ErrX=1.
%          - Matrix of the second set of control points (the "reference")
%            [X, Y, ErrX, ErrY]. If ErrY is not provided, then
%            will assume ErrY=ErrX. If ErrX is not provided then
%            ErrY=ErrX=1.
%            The reference coordinates are used as the "independent"
%            variable in the least square problem.
%          * Cell array indicating which parameters to fit.
%            The cell array should contain pairs of arguments,
%            Alternatively, seperate, ...,key,val, argument pairs.
%            Each pair constructed from a fit type and its parameters.
%            The existence of a fit type will apply it to the fit,
%            while the argument following the fit type indicate its
%            forced parameters.
%            Another option is a structure with the appropriate fields.
%            The possible fit types are:
%            'Shift' - Linear shift. If followd by an empty matrix or
%                      [NaN NaN] then will attempt to fit X and Y shift.
%                      If Folloed by [ShiftX, ShiftY] then these shifts
%                      will be forced.
%            'Rot'   - Rotation. If followd by an empty matrix or
%                      NaN matrix then will attempt to fit a rotation
%                      matrix.
%                      If the matrix contains numbers than the rotation
%                      matrix will be forced.
%            'ParAng1' - Parallactic angle stretch. Followed by
%                      a matrix which number of lines is like the number
%                      of tie points (e.g., reference stars) and 2 or 3
%                      columns. If two columns [ParAng, AM] then will
%                      attempt to fit a term of DX = Coef.*AM.*sin(ParAng)
%                      and DY = Coef.*AM.*cos(ParAng).
%                      Note that Coef is fitted simultanously to the
%                      X and Y axes.
%                      If three columns [ParAng, AM, Coef] then will
%                      force aterm of DX = Coef.*AM.*sin(ParAng)
%                      and DY = Coef.*AM.*cos(ParAng).
%            'ParAng2' - Second parallactic angle stretch (i.e.,
%                      one to the image and the other to the reference).
%                      Like ParAng1.
%            'ColParAng1' - Color refraction (parallactic angle-color
%                      dependent stretch). Followed by
%                      a matrix which number of lines is like the number
%                      of tie points (e.g., reference stars) and 3 or 4
%                      columns. If 3 columns [ParAng, AM, Color] then will
%                      attempt to fit a term of DX = Coef.*AM.*Color.*sin(ParAng)
%                      and DY = Coef.*AM.*Color.*cos(ParAng).
%                      Note that Coef is fitted simultanously to the
%                      X and Y axes.
%                      If 4 columns [ParAng, AM, Color, Coef] then will
%                      force aterm of DX = Coef.*Color.*AM.*sin(ParAng)
%                      and DY = Coef.*Color.*AM.*cos(ParAng).
%            'ColParAng2' - Second color parallactic angle (i.e.,
%                      one to the image and the other to the reference).
%                      Like ColParAng1.
%            'PolyX' - Polynomial distortion in the X-axis equation.
%                      If empty then do not fit.
%                      If six columns [DegX X0, NormX, DegY, Y0, NormY]
%                      then will attempt to fit:
%                      Coef_i*((X-X0_i)./NormX_i).^DegX_i.*((Y-Y0_i)./NormY_i).^DegY_i.
%                      If seven columns [DegX X0, NormX, DegY, Y0, NormY Coef]
%                      then will force this polynomial distortion.
%            'PolyY' - Same as PolyX but for the Y-axis equation.
%            'RadPolyX'- Radial polynomial distortion in the X-axis equation.
%                      If empty then do not fit.
%                      If five columns [DegR X0, NormX, Y0, NormY]
%                      then will attempt to fit:
%                      Coef_i*R_i.^DegR where
%                      R_i = sqrt( ((X-X0_i)./NormX_i).^2 + ((Y-Y0_i)./NormY_i).^2 ).
%                      If six columns [DegR X0, NormX, Y0, NormY Coef]
%                      then will force this polynomial distortion.
%            'RadPolyY' - Same as RadPolyX but for the Y-axis equation.
%            'FunX'  - A general function for the X-axis equation.
%                      If empty then do not fit.
%                      If a single element cell array
%                      {Fun} where Fun is a function of the form
%                      Fun(X,Y,Par), and will fit Coef.*Fun.
%                      If two elements cell array {Fun,Par} and
%                      if three elements cell array {Fun, Par, Coef}
%                      then will force Coef.*Fun on the X equation.
%            'FunY'  - Same As FunX but for the y-axis equation.
% Output : - Structure containing the best fit information.
%            Note that the RMS is in units of the reference image
%            coordinate units.
% See also: fit_affine2d_s.m
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Ref = rand(100,2).*1024;
%          Cat(:,1) =  Ref(:,1).*cosd(1)+Ref(:,2).*sind(1) + 10 + 0.01.*((Ref(:,1)-500)./500).^2;
%          Cat(:,2) = -Ref(:,1).*sind(1)+Ref(:,2).*cosd(1) + 20;
%          [FitRes]=fit_general2d_ns(Cat,Ref);
%          [FitRes]=fit_general2d_ns(Cat,Ref,{'Shift',[],'Rot',[],'PolyX',[2 500 500 2 500 500]});
%          Ref = rand(100,2).*1024;
%          Cat(:,1) =  Ref(:,1).*cosd(1)+Ref(:,2).*sind(1) + 10 + 
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==2),
    Pars = {'Shift',[],'Rot',[]};
elseif (nargin==3),
    if (isstruct(varargin{1})),
        Pars = struct2keyvalcell(varargin{1});
    else
        Pars = varargin{1};
    end
elseif (nargin>3),
    Pars = varargin;
else
    error('Illegal number of input arguments');
end
    


[Ncat,Ncatcol] = size(Cat);
[Nref,Nrefcol] = size(Ref);

if (Ncat~=Nref),
    error('Number of points in Ref and Cat must be identical');
end
N = Ncat;

% make sure Ref and Cat has four column each [X,Y,ErrX, ErrY]
if (Ncatcol==2),
    Cat = [Cat, ones(size(Cat))];
elseif (Ncatcol==3),
    Cat = [Cat, Cat(:,3)];
else
    % do nothing
end

if (Nrefcol==2),
    Ref = [Ref, ones(size(Ref))];
elseif (Nrefcol==3),
    Ref = [Ref, Ref(:,3)];
else
    % do nothing
end

    
% control points
Xcat     = Cat(:,1);
Ycat     = Cat(:,2);
Xref     = Ref(:,1);
Yref     = Ref(:,2);
ErrXcat  = Cat(:,3);
ErrYcat  = Cat(:,4);
ErrXref  = Ref(:,3);
ErrYref  = Ref(:,4);

% combine errors of Ref and Cat
ErrX     = sqrt(ErrXcat.^2 + ErrXref.^2);
ErrY     = sqrt(ErrYcat.^2 + ErrYref.^2);


X = 0;
Y = 0;
% add additional terms
H       = zeros(2.*N,0);
Nterms = numel(Pars);
TranS.Shift      = [];
TranS.Rot        = [1 0;0 1];
TranS.ParAng1    = [NaN NaN];
TranS.ParAng2    = [NaN NaN];
TranS.ColParAng1 = [NaN NaN];
TranS.ColParAng2 = [NaN NaN];
TranS.PolyX      = [];
TranS.PolyY      = [];
TranS.RadPolyX   = [];
TranS.RadPolyY   = [];
TranS.FunX       = {};
TranS.FunY       = {};
Ind = 0;
for Iterms=1:2:Nterms-1,
    switch lower(Pars{Iterms})
        case 'shift'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % fit / add to design matrix 
                H   = [H, [ones(N,1), zeros(N,1); zeros(N,1), ones(N,1)]];
                Ind = Ind + 2;
                PS.(Pars{Iterms}) = [Ind-1, Ind];
            else
                % force / subtract from ref coordinates
                X    = X + Pars{Iterms+1}(1);
                Y    = Y + Pars{Iterms+1}(2);
                
                % Xref = Xref - Pars{Iterms+1}(1);
                % Yref = Yref - Pars{Iterms+1}(2);
            end
        case 'rot'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % fit / add to design matrix 
                H   = [H, [[Xcat, Ycat, zeros(N,2)]; [zeros(N,2), Xcat, Ycat]]];
                Ind = Ind + 4;
                PS.(Pars{Iterms}) = Ind-(3:-1:0);
            else
                % force / subtract from ref coordinates
                X    = X + (Pars{Iterms+1}(1,1).*Xcat + Pars{Iterms+1}(1,2).*Ycat);
                Y    = Y + (Pars{Iterms+1}(2,1).*Xcat + Pars{Iterms+1}(2,2).*Ycat);
                
                %Xref = Xref - (Pars{Iterms+1}(1,1).*Xcat + Pars{Iterms+1}(1,2).*Ycat);
                %Yref = Yref - (Pars{Iterms+1}(2,1).*Xcat + Pars{Iterms+1}(2,2).*Ycat);
            end
        case 'parang1'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                if (size(Pars{Iterms+1},2)==2),
                    % fit / add to design matrix 
                    H = [H, [Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1)); Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1))]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % force / subtract from ref coordinates
                    X    = X + Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    Y    = Y + Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                    
                    %Xref = Xref - Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    %Yref = Yref - Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                end
            end
                
        case 'parang2'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                if (size(Pars{Iterms+1},2)==2),
                    % fit / add to design matrix 
                    H = [H, [Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1)); Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1))]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % force / subtract from ref coordinates
                    X    = X + Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    Y    = Y + Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                    
                    %Xref = Xref - Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    %Yref = Yref - Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                end
            end
        case 'colparang1'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                if (size(Pars{Iterms+1},2)==3),
                    % fit / add to design matrix 
                    H = [H, [Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1)); Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1))]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % force / subtract from ref coordinates
                    X    = X + Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    Y    = Y + Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                    
                    %Xref = Xref - Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    %Yref = Yref - Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                end
            end
        case 'colparang2'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                if (size(Pars{Iterms+1},2)==3),
                    % fit / add to design matrix 
                    H = [H, [Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1)); Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1))]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % force / subtract from ref coordinates
                    X    = X + Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    Y    = Y + Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                    
                    %Xref = Xref - Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*sin(Pars{Iterms+1}(:,1));
                    %Yref = Yref - Pars{Iterms+1}(:,4).*Pars{Iterms+1}(:,3).*Pars{Iterms+1}(:,2).*cos(Pars{Iterms+1}(:,1));
                end
            end            
        case 'polyx'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                % [DegX X0, NormX, DegY, Y0, NormY]
                Npoly = size(Pars{Iterms+1},1);
                DegX  = Pars{Iterms+1}(:,1).';
                X0    = Pars{Iterms+1}(:,2).';
                NormX = Pars{Iterms+1}(:,3).';
                DegY  = Pars{Iterms+1}(:,4).';
                Y0    = Pars{Iterms+1}(:,5).';
                NormY = Pars{Iterms+1}(:,6).';
                if (size(Pars{Iterms+1},2)==6),
                    % fit / add to design matrix 
                    H = [H, [bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),DegX) .* ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),DegY);zeros(N,Npoly)]];
                    Ind = Ind + Npoly;
                    PS.(Pars{Iterms}) = Ind - (Npoly-1:-1:0);
                else
                    % assuming 7 columns
                    % force / subtract from ref coordinates
                    P = bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),DegX) .* ...
                                  bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),DegY);
                    P = sum(bsxfun(@times,Pars{Iterms+1}(:,7).',P),2);
                    X = X + P;
                    %Xref = Xref - P;
                end
            end
        case 'polyy'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1}(:))) ),
                % ignore
            else
                % [DegX X0, NormX, DegY, Y0, NormY]
                Npoly = size(Pars{Iterms+1},1);
                DegX  = Pars{Iterms+1}(:,1).';
                X0    = Pars{Iterms+1}(:,2).';
                NormX = Pars{Iterms+1}(:,3).';
                DegY  = Pars{Iterms+1}(:,4).';
                Y0    = Pars{Iterms+1}(:,5).';
                NormY = Pars{Iterms+1}(:,6).';
                if (size(Pars{Iterms+1},2)==6),
                    % fit / add to design matrix 
                    H = [H, [zeros(N,Npoly);...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),DegX) .* ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),DegY)]];
                    Ind = Ind + Npoly;
                    PS.(Pars{Iterms}) = Ind - (Npoly-1:-1:0);
                else
                    % assuming 7 columns
                    % force / subtract from ref coordinates
                    P = bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),DegX) .* ...
                                  bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),DegY);
                    P = sum(bsxfun(@times,Pars{Iterms+1}(:,7).',P),2);
                    Y = Y + P;
                    %Yref = Yref - P;
                end
            end  
        case 'radpolyx'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1})) ),
                % ignore
            else
                % [DegR X0, NormX, Y0, NormY]
                Npoly = size(Pars{Iterms+1},1);
                DegR  = Pars{Iterms+1}(:,1).';
                X0    = Pars{Iterms+1}(:,2).';
                NormX = Pars{Iterms+1}(:,3).';
                Y0    = Pars{Iterms+1}(:,4).';
                NormY = Pars{Iterms+1}(:,5).';
                if (size(Pars{Iterms+1},2)==5),
                    % fit / add to design matrix 
                    P = sqrt(bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),2) + ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),2));
                    P = bsxfun(@power,P,DegR);
                    H = [H, [P; zeros(N,Npoly)]];
                    Ind = Ind + Npoly;
                    PS.(Pars{Iterms}) = Ind - (Npoly-1:-1:0);
                else
                    % assuming 7 columns
                    % force / subtract from ref coordinates
                    P = sqrt(bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),2) + ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),2));
                    P = sum(bsxfun(@times,Pars{Iterms+1}(:,6).', bsxfun(@power,P,DegR)),2);
                    X = X + P;
                    %Xref = Xref - P;
                end
            end 
            
        case 'radpolyy'
            if (isempty(Pars{Iterms+1}) || all(isnan(Pars{Iterms+1})) ),
                % ignore
            else
                % [DegR X0, NormX, Y0, NormY]
                Npoly = size(Pars{Iterms+1},1);
                DegR  = Pars{Iterms+1}(:,1).';
                X0    = Pars{Iterms+1}(:,2).';
                NormX = Pars{Iterms+1}(:,3).';
                Y0    = Pars{Iterms+1}(:,4).';
                NormY = Pars{Iterms+1}(:,5).';
                if (size(Pars{Iterms+1},2)==5),
                    % fit / add to design matrix 
                    P = sqrt(bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),2) + ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),2));
                    P = bsxfun(@power,P,DegR);
                    H = [H, [zeros(N,Npoly); P]];
                    Ind = Ind + Npoly;
                    PS.(Pars{Iterms}) = Ind - (Npoly-1:-1:0);
                else
                    % assuming 7 columns
                    % force / subtract from ref coordinates
                    P = sqrt(bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Xcat,X0),NormX),2) + ...
                             bsxfun(@power,bsxfun(@rdivide,bsxfun(@minus,Ycat,Y0),NormY),2));
                    P = sum(bsxfun(@times,Pars{Iterms+1}(:,6).', bsxfun(@power,P,DegR)),2);
                    Y = Y + P;
                    %Yref = Yref - P;
                end
            end 
        case 'funx'
            % General function on X equation
            Np = numel(Pars{Iterms+1});
            if (Np==0),
                % ignore
            else
                if (Np==1),
                    % fit
                    Fun = Pars{Iterms+1}{1};
                    H = [H, [Fun(Xcat,Ycat); zeros(N,1)]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                elseif (Np==2),
                    Fun = Pars{Iterms+1}{1};
                    ParFun = Pars{Iterms+1}{2};
                    H = [H, [Fun(Xcat,Ycat,ParFun); zeros(N,1)]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % Np>2 - force
                    Fun = Pars{Iterms+1}{1};
                    ParFun = Pars{Iterms+1}{2};
                    X = X + Pars{Iterms+1}{3}.*Fun(Xcat,Ycat,ParFun);
                    %Xref = Xref - Pars{Iterms+1}{3}.*Fun(Xcat,Ycat,ParFun);
                end
            end
            
        case 'funy'
            % General function on Y equation
            Np = numel(Pars{Iterms+1});
            if (Np==0),
                % ignore
            else
                if (Np==1),
                    % fit
                    Fun = Pars{Iterms+1}{1};
                    H = [H, [zeros(N,1); Fun(Xcat,Ycat)]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                elseif (Np==2),
                    Fun = Pars{Iterms+1}{1};
                    ParFun = Pars{Iterms+1}{2};
                    H = [H, [zeros(N,1); Fun(Xcat,Ycat,ParFun)]];
                    Ind = Ind + 1;
                    PS.(Pars{Iterms}) = Ind;
                else
                    % Np>2 - force
                    Fun = Pars{Iterms+1}{1};
                    ParFun = Pars{Iterms+1}{2};
                    Y = Y + Pars{Iterms+1}{3}.*Fun(Xcat,Ycat,ParFun);
                    %Yref = Yref - Pars{Iterms+1}{3}.*Fun(Xcat,Ycat,ParFun);
                end
            end
         
        otherwise
            error('Unknown model option');
    end
end
                     
if (isempty(H)),
    % H is empty - do not fit
    FitRes = [];
else
    % fit
    %[ParX,ParErrX] = lscov(Hx,Xref,ErrX.^-2);
    %[ParY,ParErrY] = lscov(Hy,Yref,ErrY.^-2);
    
    XYref = [Xref;Yref];
    Err   = [ErrX;ErrY];
    % good flag
    Fgx   = ~isnan(Xref) & ~isnan(Yref);
    Fgy   = Fgx; %~isnan(Yref);
    Fg    = [Fgx;Fgy];
    H     = H(Fg,:);
    XYref = XYref(Fg);
    Err   = Err(Fg);
    
    InPar.MaxIter = 4;
    for I=1:1:InPar.MaxIter,
        N     = size(H,1).*0.5;

        [Par,ParErr]    = lscov(H,XYref,Err.^-2);    
        Hx = H(1:N,:);
        Hy = H(N+1:end,:);

        % residuals
        FitRes.ResidX = Xref - Hx*Par;
        FitRes.ResidY = Yref - Hy*Par;
        FitRes.Resid  = [FitRes.ResidX; FitRes.ResidY];
        FitRes.ResidR = sqrt(FitRes.ResidX.^2 + FitRes.ResidY.^2);

        if (I<InPar.MaxIter),
            InPar.ClipPars = {'Method','StdP','Clip',[2 2]};

            [~,~,FlagGood] = clip_resid(FitRes.ResidR,InPar.ClipPars{:});
            Xref = Xref(FlagGood);
            Yref = Yref(FlagGood);
            ErrX = ErrX(FlagGood);
            ErrY = ErrY(FlagGood);
            XYref = [Xref;Yref];
            Err   = [ErrX;ErrY];
            H    = H([FlagGood;FlagGood],:);
        end
    end
    %hist(FitRes.Resid,100);
    
    FitRes.Par    = Par;
    FitRes.ParErr = ParErr;
    FitRes.PS     = PS;

    % rms
    FitRes.Xrms   = std(FitRes.ResidX);
    FitRes.Yrms   = std(FitRes.ResidY);
    FitRes.rms    = std(FitRes.Resid);
    % chi2
    FitRes.Chi2   = sum(FitRes.ResidX./ErrX) + sum(FitRes.ResidY./ErrY);
    FitRes.N      = 2.*N;
    FitRes.Nsrc   = N;
    FitRes.Npar   = numel(Par);
    FitRes.Ndof   = FitRes.N - FitRes.Npar;

    FitRes.tdata.Par = Par;
    FitRes.tdata.ParErr = ParErr;
    FitRes.tdata.PS     = PS;

    % transformation
    %FitRes.Tran          = affine2d([[ParX(1) -ParX(2) ParX(3)]; [ParY(1), +ParY(2), ParY(3)]; 0 0 1].');

    % populate parameters
    IndX = 0;
    IndY = 0;
    for Iterms=1:2:Nterms-1,
        %TranS.(Pars{Iterms}) = Par(PS.(Pars{Iterms}));
        switch lower(Pars{Iterms})
            case 'shift'
                 TranS.(Pars{Iterms}) = Par(PS.(Pars{Iterms}));
            case 'rot'
                TranS.(Pars{Iterms}) = [Par(PS.(Pars{Iterms})(1)), Par(PS.(Pars{Iterms})(2)); Par(PS.(Pars{Iterms})(3)), Par(PS.(Pars{Iterms})(4))];
            case {'parang1','parang2'}
                TranS.(Pars{Iterms}) = [Pars{Iterms+1}, Par(PS.(Pars{Iterms}))];
            case {'polyx','polyy','radpolyx','radpolyy'}
                TranS.(Pars{Iterms}) = [Pars{Iterms+1}, Par(PS.(Pars{Iterms}))];
            case {'funx','funy'}
                TranS.(Pars{Iterms}) = [Pars{Iterms+1}, Par(PS.(Pars{Iterms}))];
                
        end
    end

    %[~,T] = coo_trans([],true,TranS);
    Fun   = @(XY,T) coo_trans(XY,true,TranS);
    FitRes.Tran  = maketform('custom', 2, 2, [], Fun, TranS);
    FitRes.TranS = TranS;

end
