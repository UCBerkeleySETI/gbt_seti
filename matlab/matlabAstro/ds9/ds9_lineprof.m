function Vec=ds9_lineprof(ImageName,varargin)
%--------------------------------------------------------------------------
% ds9_lineprof function                                                ds9
% Description: Interactive examination of vector in a FITS image displayed
%              in ds9. The program display an image in ds9 and then prompt
%              the user to click on two points in the image. Than the
%              program reads from the image a vector between the two points
%              and return/display the vector.
%              THIS FUNCTION WILL BE REMOVED - USE ds9_exam.m
% Input  : - FITS image file name (string).
%            Alternatively, this can be a matrix that will be saved as
%            a temporary FITS file and will be displayed in ds9.
%            If empty matrix than use existing image in the ds9 display.
%            Default is empty.
%          * Arbitrary number of pairs of ...,key,val,... arguments.
%            The following keywords are available:
%            'KeyType' - The key the user need to click on in order to mark
%                        one of the vector ends. This parameter can be
%                        a string containing the name of any keyboard key
%                        e.g., 'v' or 'Right' ot 'x', or one of the 
%                        following strings.
%                        'key'   - Any keyboard key.
%                        'any'   - Any keyboard key or mouse left click.
%                        'vxyij' - will react differently for each letter,
%                              according to the scheme described in 'VecType'.
%                        Alternatively this can be a 4 column matrix
%                        containing the following columns:
%                        [Xstart, Ystart, Xend, Yend].
%                        Default is 'vxyij'.
%            'Behav'   - Script behaviour:
%                        'q' - Quit the script only when the user pressed
%                              'q'.
%                        number - If this is a number (N>0), then the user
%                              will be prompted N times to mark a vector.
%                        Default is 1.
%            'VecType' - This can be one of the following types:
%                        'v' - Vector defined by two points.
%                        'x' - Vector along the x-axis where the vector
%                              end points are defined by the mean value of
%                              y between the two clicks.
%                        'y' - Like 'x', but for the 'y' axis.
%                        'i' - Vector defined by a single point along
%                              the y axis where the center of the vector
%                              is the clicked position and the length of
%                              the vector is defined by 'VecLength'.
%                        'j' - Like 'i', but for the x axis.
%                        'vxyij' - will react differently for each letter,
%                              according to the above scheme.
%                        Default is 'vxyij'.
%            'VecLength'- Vector half length (used when VecType is 'i' or 'j').
%                        Default is 50.
%            'VecWidth'- Vector width. If >1, then will
%                        return the mean or median of the values along the
%                        axis perpendicular to trhe vector.
%                        Default is 1.
%                        If the ImageName is [] then this parameter is
%                        set to 1.
%            'VecMean' - The averaging function along the vector width.
%                        {'sum' | 'mean' | 'median' | 'std' | 'min' | 'max'}
%                        Default is 'median'.
%            'VecInterp'- Interpolation method in calculating values along
%                        the vector. See interp1.m for options.
%                        Default is 'linear'.
%            'ReadMethod'- The method to use in order to read the pixels
%                        values:
%                        'fits' - reading the fits file if available.
%                                 Default.
%                        'ds9'  - reading the pixels directly from ds9.
%            'FitsPar' - Cell array of parameters to pass to the firstead.m
%                        program. Default is {}.
%            'Frame'   - ds9 frame (see ds9_disp.m for options).
%                        Default is 1.
%            'DispPar' - Cell array of parameters to pass to ds9_disp.m
%                        Default is {}.
%            'Plot'    - Plot the line profile in a matlab figure.
%                        Options are:
%                        'none'  - do not plot (default).
%                        'plot'  - plot line profile.
%            'PlotPar' - Cell array of additional parameters to pass to
%                        the plot function.
%                        Default is {'k-','LineWidth',1}.
% Output : - Structure or structure array (for multiple vectors) of the
%            vector properties. This is including the following fields:
%            .Start - [X,Y] start position of the vector.
%            .End   - [X,Y] end position of the vector.
%            .Vec   - Vector values.
%            .VecR, .VecX, .VecY - index of pix along the vector, x, and y.
%            .VecMean, .VecWidth, .VecLength, .VecType (from the input).
% See also: ds9_exam.m
% Tested : Matlab 2011a
%     By : Eran O. Ofek                    Jan 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Vec=ds9_lineprof('PTF201001131475_2_o_14235_00.w.fits');
% Reliable: 2
%--------------------------------------------------------------------------
CooType = 'image';

Def.ImageName = [];
if (nargin==0),
   ImageName = Def.ImageName;
end

DefV.KeyType     = 'v';
DefV.Behav       = 1;
DefV.VecType     = 'v';
DefV.VecLength   = 50;
DefV.VecWidth    = 1;
DefV.VecMean     = 'median';
DefV.VecInterp   = 'linear';
DefV.ReadMethod  = 'fits';
DefV.FitsPar     = {};
DefV.Frame       = 1;
DefV.DispPar     = {};
DefV.Plot        = 'none';
DefV.PlotPar     = {'k-','LineWidth',1};
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


PlotH = [];  % plot handle

Vec = [];
% Read Image
% Mat will be empty if need to read it from ds9.
if (isempty(ImageName)),
    % override ReadMethod
    InPar.ReadMethod = 'ds9';
    Mat = [];
else
   if (isnumeric(ImageName)),
      % save image as a temporary FITS file
      Mat = ImageName;
      ImageName = tempname;
      fitswrite(Mat,ImageName);
   elseif (ischar(ImageName)),
      Mat = fitsread(ImageName,InPar.FitsPar{:});
   end

   ds9_disp(ImageName,InPar.Frame,InPar.DispPar{:});
end

if (isnumeric(InPar.Behav)),
   LoopLimit    = InPar.Behav;
   LoopLimitKey = [];
   PrintedMessage = sprintf('');
elseif (ischar(InPar.Behav))
   switch lower(InPar.Behav)
    case 'q'
       LoopLimit = Inf;
       LoopLimitKey = 'q';
       PrintedMessage = sprintf('Type ''q'' to exit\n');
    otherwise
       error('Unknown Behav option');
   end
else
   error('Unknown Behav type');
end

if (isnumeric(InPar.KeyType)),
   % loop over all data points
   Nvec = size(InPar.KeyType,1);
   VecPos = zeros(Nvec,4);
   for Ivec=1:1:Nvec,
      % VecPos = [Xstart, Ystart, Xend, Yend].
      VecPos(Ivec,:) = InPar.KeyType(LoopInd,:);
   end
else
   % assuming KeyType is character

   IsExit  = false;
   LoopInd = 0;
   VecPos  = zeros(0,4);
   
   while (LoopInd<LoopLimit && ~IsExit)
      LoopInd = LoopInd + 1;
      fprintf('%s',PrintedMessage);
      
      switch lower(InPar.VecType)
       case {'i','j'}
          [CooX1,CooY1,Value1,RetKey1,IsExit] = ds9_lineprofe_user_click(InPar.KeyType,CooType,IsExit,'');
          CooX2 = CooX1;
          CooY2 = CooY1;
          Value2 = Value1;
          RetKey2 = RetKey1;
       case {'v','x','y'}
          [CooX1,CooY1,Value1,RetKey1,IsExit] = ds9_lineprofe_user_click(InPar.KeyType,CooType,IsExit,'first');

          if (IsExit),
             % skip next point
          else
             % get second point
             [CooX2,CooY2,Value2,RetKey2,IsExit] = ds9_lineprofe_user_click(InPar.KeyType,CooType,IsExit,'second');
          end
       case 'vxyij'
           [CooX1,CooY1,Value1,RetKey1,IsExit] = ds9_lineprofe_user_click(InPar.KeyType,CooType,IsExit,'first')
           if (IsExit),
               % skip next point
           else
               switch lower(RetKey1)
                   case {'x','y','i','j'}
                       % skip next point
                   case 'v'
                       % get second point
                       [CooX2,CooY2,Value2,RetKey2,IsExit] = ds9_lineprofe_user_click(InPar.KeyType,CooType,IsExit,'second');
                   otherwise
                       error('Unknown VecType option');
               end
           end
           
           
       otherwise
          error('Unknown VecType option');
      end

      if (IsExit),
         fprintf('Exit interactive mode\n');
      end

      % VecPos = [Xstart, Ystart, Xend, Yend].
      
      if (~IsExit),
    	 switch lower(InPar.VecType)         
          case 'v'
             VecPos(LoopInd,:) = [CooX1, CooY1, CooX2, CooY2];
          case 'x'
 	         CooY1 = mean([CooY1;CooY2]);
             CooY2 = CooY1;
          case 'y'
 	         CooX1 = mean([CooX1;CooX2]);
             CooX2 = CooX1;
          case 'i'
 	         CooX2 = CooX1;
             CooY1 = CooY1 - InPar.VecLength;
             CooY2 = CooY2 + InPar.VecLength;
          case 'j'
 	         CooY2 = CooY1;
             CooX1 = CooX1 - InPar.VecLength;
             CooX2 = CooX2 + InPar.VecLength;
          otherwise
	     error('Unknown VecType option');
         end
         VecPos(LoopInd,:) = [CooX1, CooY1, CooX2, CooY2];
      
         % get vector
         Ivec = LoopInd;
         if (isempty(Mat)),
            if (InPar.VecWidth>1),
                fprintf('In this mode set VecWidth=1\n');
                InPar.VecWidth = 1;
            end
            Xmin = floor(min(VecPos(Ivec,1),VecPos(Ivec,3)));
            Ymin = floor(min(VecPos(Ivec,2),VecPos(Ivec,4)));
            Xmax = ceil(max(VecPos(Ivec,1),VecPos(Ivec,3)));
            Ymax = ceil(max(VecPos(Ivec,2),VecPos(Ivec,4)));            
            BoxCorner = [Xmin, Ymin, Xmax-Xmin, Ymax-Ymin];
            [MatVal,MatX,MatY]=ds9_getbox(BoxCorner,'corner',CooType);
            Start = [1 1];
            End   = [size(MatVal,2), size(MatVal,1)];
         else
            MatVal = Mat;
            [MatX,MatY] = meshgrid( (1:size(MatVal,2)),...
			                        (1:size(MatVal,1)) );
            Start = VecPos(Ivec,1:2);
            End   = VecPos(Ivec,3:4);
         end

         %--- Calculate vector values ---
         [Prof,VecR,VecX,VecY]=lineprof(MatVal,Start,End,...
                                        InPar.VecWidth,1,...
                                        InPar.VecMean,InPar.VecInterp);

          Vec(Ivec).Start    = Start;
          Vec(Ivec).End      = End;
          Vec(Ivec).Vec      = Prof;
          Vec(Ivec).VecR     = VecR;
          Vec(Ivec).VecX     = VecX;
          Vec(Ivec).VecY     = VecY;

          Vec(Ivec).VecMean   = InPar.VecMean;
          Vec(Ivec).VecWidth  = InPar.VecWidth;
          Vec(Ivec).VecLength = InPar.VecLength;
          Vec(Ivec).VecType   = InPar.VecType;
          
          switch lower(InPar.Plot)
              case 'none'
                  % do nothing
              case 'plot'
                  error('plot does not work yet');
                  plot(Vec(Ivec).VecR,Vec(Ivec).Vec,InPar.PlotPar{:});
                  H = xlabel('Radius [pix]');
                  set(H,'FontSize',16);
                  H = ylabel('Value');
                  set(H,'FontSize',16);
                  drawnow;
                  pause(0.1);
                  
              otherwise
                  error('Unknown Plot option');
          end
              
      end
   end
end






%----------------------------------------------------------------------------
function [CooX,CooY,Value,RetKey,IsExit]=ds9_lineprofe_user_click(KeyType,CooType,IsExit,Number);
%----------------------------------------------------------------------------

switch lower(KeyType)
 case {'key','any','mouse'}
    switch lower(KeyType)
     case 'key'
        fprintf('Mark the %s vector point in the ds9 display using any keybord key\n',Number);
     case 'any'
        fprintf('Mark the %s vector point in the ds9 display using any keybord key or mouse left click\n',Number);
     otherwise
        error('Bug in ds9_exvec.m');
    end
    [CooX,CooY,Value,RetKey]=ds9_getcoo(1,CooType,KeyType);
    
    switch lower(RetKey{1})
     case 'q'
        IsExit = true; 

     otherwise
        if (IsExit),
	   IsExit = true;
        else
	   IsExit = false;
        end
    end
 otherwise
    % respond to specific key
    if (~IsExit),
       fprintf('Mark the %s vector point in the ds9 display using the ''%s'' key\n',Number,KeyType);

       [CooX,CooY,Value,RetKey]=ds9_getcoo(1,CooType,'key');
       
    end
    
    while (isempty(findstr(RetKey{1},KeyType)) && ~IsExit)
       switch lower(RetKey{1})
        case 'q'
           IsExit = true;
        otherwise
           if (IsExit),
	          IsExit = true;
           else
	          IsExit = false;
           end
       end
       if (~IsExit),
 	      fprintf('Mark the %s vector point in the ds9 display using the ''%s'' key\n',Number,KeyType);
          [CooX,CooY,Value,RetKey]=ds9_getcoo(1,CooType,'key');
       end
    end
end
