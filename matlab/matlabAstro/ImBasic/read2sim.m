function [Struct,ListCell,Cube]=read2sim(Input,varargin)
%-----------------------------------------------------------------------------
% read2sim function                                                   ImBasic
% Description: Read a list of images into a structure array. A flexible
%              utility that allows to generate a structure array of images
%              from one of the following inputs: a matrix; cell of matrices;
%              a string with wild cards (see sdir for flexibility);
%              a file containing list of images (see create_list.m);
%              a cell array of file names; or a structure array with
%              image names.
%              see read2cat.m for the analog function that works with
%              catalogs.
%              THIS FUNCTION IS OBSOLOTE use images2sim.m instead.
% Input  : - Input. This can be one of the followings:
%            (1) A matrix.
%            (2) A cell array of matrices.
%            (3) A string containing a file name with optional wild
%                cards and or ranges.
%                e.g., 'lred*.fits'; 'lred0[15-28].fits'
%                Files can be of many types and formats.
%            (4) A string starting with '@'. In this case the rest of
%                the string contains a file which contains in each line
%                a file name.
%            (5) A cell array in which each element is a string
%                containing a file name. e.g., {'a.fits';'b.fits'}.
%            (6) A structure array which contains a field named
%                'ImageFileName'.
%                Note that the structure will be used only for the
%                file names. You can pass additional info using
%                the 'Struct' option explained below.
%            (7) A structure array which contains a field named
%                'Im'. In this case it will be returned as output.
%            The files/matrices in all these options will be read into
%            a structure array. If the images are read from a
%            fits image then the header of each image will be read too.
%          * Arbitrary number of pairs of input arguments ...,key,val,...
%            The following keywords are supported:
%            'Count'  - {false|true}. If true then the program will not
%                       read the files into a structure array and
%                       will only return a counter of the number of
%                       available images. Default is false.
%            'Index'  - This is a vector of integers specifying the
%                       index of the images to read. 
%                       Default is empty matrix (i.e., read all).
%            'Struct' - A structure array in which the number of
%                       elements is identical to the number of images
%                       to read. If provided then the .Im field will
%                       be added into this existing structure array.
%                       Default is empty matrix (i.e., no structure).
%            'CCDSEC' - Section of CCD to read. This can be a vector
%                       of [Xmin, Xmax, Ymin, Ymax], or a string
%                       which contains a FITS header keyword which
%                       contains the data section to read (will work
%                       only if the input images are FITS files).
%                       If empty then read the entire image.
%                       Default is empty.
%            'Format' - Input images format. Supported formats are:
%                       'FITS' - Files are in FITS image format (default).
%                       'mat'  - mat files each containins a matrix.
%                       'smat' - mat files each containing a structure
%                       'ssmat'- A single mat file containing a
%                                structure array.
%                       'imread'- Will use the imread.m function to read
%                                the images.
%            'Extension' - Index of FITS file extension. Default is 1.
%            'ImreadOpt' - Cell array of additional optional parameters
%                       to pass to imread.m. Default is empty cell.
% Output : - A structure array of the images.
%            The structure always contains a field named 'Im' which
%            contain the image in a matrix format.
%            Additional fields may include:
%            .Header - Cell array of Nx3 of image header.
%            .ImageFileName - name of file from which the image was read.
%            .CCDSEC - the read section in coordinates of the original
%                      image. If empty then equal to original image.
%          - Cell array of images file names. If not available then will
%            return an empty matrix.
%          - A matrix cube of all images ordered in a cube.
%            This is possible only if all the images have the same size.
%            If some of the images have different sizes then a empty
%            matrix will be returned.
% See also: sim2cube.m, sim2file.m, read2cat.m
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Aug 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Struct,ListCell,Cube]=read2sim('try*.fits','CCDSEC',[1 10 1 20]);
%          Struct=read2sim('@list);
%          Struct=read2sim({rand(10,10)});
%          Struct=read2sim({'a.fits','b.fits'});
%          Struct=read2sim({'a.fits','b.fits'},'Struct',A);
%          Struct=read2sim('S.mat','Format','ssmat');
%          Struct=read2sim({'S1.mat','S2.mat'},'Format','smat');
%          Counter=read2sim('*.fits','Count',true);
%          Struct=read2sim('*.fits','Index',[2 3]);
%          Struct=read2sim({rand(2,2), 2, 3},'Index',[1 3]);
% Reliable: 2
%-----------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%InFileField = 'InputFileName';


DefV.Count     = false;
DefV.Index     = [];
DefV.Struct    = [];
DefV.CCDSEC    = [];
DefV.Format    = 'FITS';
DefV.Extension = 1;
DefV.ImreadOpt = {};

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~isempty(InPar.CCDSEC)),
    MessageID = 'read2sim:WCS';
    warning(MessageID,'WCS is not updated according to CCDSEC');
end


if (ischar(Input)),
   % deal with options #3 and #4
   [~,Input] = create_list(Input,NaN);
   
   if (~isempty(InPar.Index)),
       Input = Input(InPar.Index);
       InPar.Index = [];
   end
   % need to read images from ListCell
end


if (isnumeric(Input)),
   % deal with option #1
   % single matrix
   Input = {Input};
   if (~isempty(InPar.Index)),
       % do nothing
       %MessageID = 'read2sim:Index';
       %warning(MessageID,'Index keyword value is ignored');
   end
end

if (iscell(Input)),
   if (isnumeric(Input{1})),
      % deal with option #2
      % list of matrices
      ListCell = [];
      if (ischar(InPar.CCDSEC)),
	     error('CCDSEC can not be a string when Input is a matrix');
      end
      if (~isempty(InPar.CCDSEC)),
          if (~isempty(InPar.Index)),
              Input = Input(InPar.Index);
              InPar.Index = [];
              Nim   = length(Input);
          end
      
          MatImage = cell(Nim,1);
	      for Iim=1:1:length(Input),
    	      MatImage{Iim} = sub_image(Input{Iim},InPar.CCDSEC,[],'boundry');
          end
      else
          if (~isempty(InPar.Index)),
              Input = Input(InPar.Index);
              InPar.Index = [];
              Nim   = length(Input);
          end
          Nim = length(Input);
          MatImage = cell(1,Nim);
          for Iim=1:1:Nim,
             MatImage{Iim} = Input{Iim};
          end
      end
      
      Struct = cell2struct(MatImage,ImageField,1);
      Nst    = length(Struct);
      [Struct(1:1:Nst).CCDSEC] = deal(InPar.CCDSEC);  % add CCDSEC to structure
   else
      % deal with option #5
      % list of images to read  
      ListCell = Input;
      if (~isempty(InPar.Index)),          
          ListCell = ListCell(InPar.Index);
          InPar.Index = [];
      end
   end
elseif (isstruct(Input)),
   if (isfield(Input,FileField)),
      % deal with option #6
      if (~isempty(InPar.Index)),
         Input = Input(InPar.Index);
         InPar.Index = [];
      end
      ListCell = {Input.(FileField)};
   end
      
   if (isfield(Input,ImageField)),
      % deal with option #7
      if (~isempty(InPar.Index)),
         Input = Input(InPar.Index);
         InPar.Index = [];
      end
      Struct   = Input;
      ListCell = [];
   end
else
   error('Unsupported Input type');
end

if (isempty(ListCell)),
   % assume that the basic structure was already constructed
   % apply CCDSEC
   Nim = length(Struct);
   
   if (InPar.Count),
       % only count the number of images
       Struct = Nim;
   else
       for Iim=1:1:Nim,
           if (~isempty(InPar.CCDSEC)),
              if (ischar(InPar.CCDSEC)),
                  [NewCellHead]       = cell_fitshead_getkey(Struct(Iim).(HeaderField),InPar.CCDSEC);
                  CCDSEC              = NewCellHead{1,2};
                  [CCDSEC]            = ccdsec_convert(CCDSEC);  % convert string to vector
               else
                   % assume InPar.CCDSEC is a vector
                   CCDSEC = InPar.CCDSEC;
               end
               % sub image
               Struct(Iim).(ImageField) = sub_image(Struct(Iim).(ImageField),CCDSEC,[],'boundry');
               Struct(Iim).CCDSEC = CCDSEC;
            end
       end     
   end
else
   Nim = length(ListCell);
   
   if (InPar.Count),
       % only count the number of images
       Struct = Nim;
   else
       % attempt to read the images
       switch lower(InPar.Format)
        case 'fits'
            for Iim=1:1:Nim,
                Image  = fitsread(ListCell{Iim},'primary',InPar.Extension);
                H      = fitsinfo(ListCell{Iim});
                Header = H.PrimaryData(InPar.Extension).Keywords;
                if (ischar(InPar.CCDSEC)),
                     [NewCellHead]       = cell_fitshead_getkey(Header,InPar.CCDSEC);
                     CCDSEC              = NewCellHead{1,2};
                     [CCDSEC]            = ccdsec_convert(CCDSEC);  % convert string to vector
                     % sub image
                     [Image]             = sub_image(Image,CCDSEC,[],'boundry');
                else
                     CCDSEC = InPar.CCDSEC;
                end

                if (~isempty(CCDSEC)),
                    [Image]             = sub_image(Image,CCDSEC,[],'boundry');
                end
                Struct(Iim).Im            = Image;
                Struct(Iim).ImageFileName = ListCell{Iim};
                Struct(Iim).Header        = Header;
                Struct(Iim).CCDSEC        = CCDSEC;
            end
        case 'mat'
            for Iim=1:1:Nim,
                Image = load2(ListCell{Iim});

                if (~isempty(InPar.CCDSEC)),
                        [Image]             = sub_image(Image,InPar.CCDSEC,[],'boundry');
                end
                Struct(Iim).Im            = Image;
                Struct(Iim).CCDSEC        = InPar.CCDSEC;
            end
        case 'smat'
            for Iim=1:1:Nim,
                ReadSt = load2(ListCell{Iim});
                if (length(ReadSt)>1),
                   error('Multiple mat files each containing a structure array are not supported');
                end
                Struct(Iim) = ReadSt;
                if (~isempty(InPar.CCDSEC)),
                    [Image]             = sub_image(Struct(Iim).(ImageField),InPar.CCDSEC,[],'boundry');
                    Struct(Iim).(ImageField) = Image;
                end
             end
        case 'ssmat'
            % a single structure array containing multiple images
            if (Nim>1),
               error('Multiple input files each containing structure array are not supported');
            end
            Struct = load2(ListCell{1});
            Nim    = length(Struct);
            for Iim=1:1:Nim,
                if (~isempty(InPar.CCDSEC)),
                    [Image]             = sub_image(Struct(Iim).(ImageField),InPar.CCDSEC,[],'boundry');
                    Struct(Iim).(ImageField) = Image;
                end
            end
        case 'imread'
            for Iim=1:1:Nim,
                Image = imread(ListCell{Iim},InPar.ImreadOpt{:});
                if (~isempty(InPar.CCDSEC)),
                    [Image]             = sub_image(Image,InPar.CCDSEC,[],'boundry');
                end
                Struct(Iim).Im            = Image;
                Struct(Iim).CCDSEC        = InPar.CCDSEC;
            end 
        otherwise
        error('Unknown Format option');
       end
   end
end


% append additional info to Struct
if (InPar.Count),
   % do nothing
   Cube = [];
else
    if (~isempty(InPar.Struct)),
       FN = fieldnames(InPar.Struct);
       Nim = length(Struct);
       for Iim=1:1:Nim,
           for If=1:1:length(FN),
               Struct(Iim).(FN{If}) = InPar.Struct(Iim).(FN{If});
           end
       end
    end


    % build the Cube
    
    if (nargout>2),
        Cube=sim2cube(Struct);
    end
end

if (~isfield(Struct,FileField)),
    EmpCell = cell(1,Nim);
    [Struct(1:1:Nim).(FileField)] = deal(EmpCell{:});
end

if (~isfield(Struct,HeaderField)),
    EmpCell = cell(1,Nim);
    [Struct(1:1:Nim).(HeaderField)] = deal(EmpCell{:});
end