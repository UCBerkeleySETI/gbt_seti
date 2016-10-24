function Files=pwget(Links,Extra,MaxGet,BaseURL)
%------------------------------------------------------------------------------
% pwget function                                                           www
% Description: Parallel wget function designed to retrieve multiple files
%              using parallel wget commands.
%              If fast communication is available, running several wget
%              commands in parallel allows almost linear increase in the
%              download speed.
%              After exceuting pwget.m it is difficult to kill it. In
%              order to stop the execuation while it is running you
%              have to create a file name 'kill_pwget' in the directory
%              in which pwget is running (e.g., "touch kill_pwget").
% Input  : - Cell array of URL file links to download.
%          - Additional string to pass to the wget command
%            e.g., '-q'. Default is empty string ''.
%          - Maxium wget commands to run in parallel.
%            Default is 5.
%          - An optional URL base to concatenate to the begining of each
%            link. This is useful if the Links cell array contains only
%            relative positions. Default is empty string ''.
%            If empty matrix then use default.
% Output : Original names of retrieved files.
% Tested : Matlab 2012a
%     By : Eran O. Ofek                    Oct 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: tic;pwget(Links,'',10);toc
% Speed  : On my computer in the Weizmann network I get the following
%          results while trying to download 20 corrected SDSS fits images:
%          MaxGet=1  runs in 83 seconds
%          MaxGet=2  runs in 41 seconds
%          MaxGet=5  runs in 19 seconds
%          MaxGet=10 runs in 9 seconds
%          MaxGet=20 runs in 6 seconds
% Reliable: 2
%------------------------------------------------------------------------------

Def.Extra   = '';
Def.MaxGet  = 5;
Def.BaseURL = '';
if (nargin==1),
   Extra   = Def.Extra;
   MaxGet  = Def.MaxGet;
   BaseURL = Def.BaseURL;
elseif (nargin==2),
   MaxGet = Def.MaxGet;
   BaseURL = Def.BaseURL;
elseif (nargin==3),
    BaseURL = Def.BaseURL;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(BaseURL)),
    BaseURL = Def.BaseURL;
end


Nlink = length(Links);
Nloop = ceil(Nlink./MaxGet);
Nget  = ceil(Nlink./Nloop);

Files = cell(Nlink,1);
Abort = false;
for Iloop=1:1:Nloop,
   Str = '';
   for Iget=1:1:Nget,
      Ind = Nget.*(Iloop-1) + Iget;
      if (Ind<=Nlink),
         if (~isempty(Str)),
             Str = sprintf('%s &',Str);
         end
 	     Str = sprintf('%s wget %s "%s%s"',Str,Extra,BaseURL,Links{Ind});
         Split=regexp(Links{Ind},'/','split');
         Files{Ind} = Split{end};
      end
   end
   % Check if user requested to kill process
   if (exist('kill_pwget','file')>0 || Abort),
       % Abort (skip wget)
       delete('kill_pwget');
       Abort = true;
   else
      system(Str);
   end
end
