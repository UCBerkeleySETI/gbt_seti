function [IndGroup,Group]=sim_group_keyval(Sim,Keys,varargin)
%--------------------------------------------------------------------------
% sim_group_keyval function                                        ImBasic
% Description: Given a structure array with image headers, and a list of
%              header keywords, construct groups of images with identical
%              keyword header values.
% Input  : - Structure array of images with header information.
%            This can be any input that is acceptable by images2sim.m
%          - Cell array of header keywords by which to group the
%            observations.
%          * Arbitrary number of pairs of ...,key,val,... arguments.
%            The following keywords are available:
%            'IsSkip'    - Vector of logicals specifing if image should
%                          be skiped (i.e., not classified into groups).
%                          In this case IndGroup.Igroup will be set
%                          to NaN.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure array with a field named Igroup, indicating the
%            group index to which the image belongs.
%            This is NaN if IsSkip is true.
%          - Structure array of possible groups. The Vals field contains
%            the values of the header keywords.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example:
% [IndGroup,Group]=sim_group_keyval(AllSimHead,InPar.SpecConfigTypeKey,'IsSkip',IsBias);
% Reliable: 2
%-----------------------------------------------------------------------------



%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.IsSkip = [];
InPar = set_varargin_keyval(DefV,'m','use',varargin{:});


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

if (isempty(InPar.IsSkip)),
    InPar.IsSkip = false(Nim,1);
end


Nkeys = numel(Keys);
Check = zeros(Nkeys,1);

Group    = struct('Vals',cell(1,0));
IndGroup = struct('Igroup',cell(1,0));
Igroup = 0;
for Iim=1:1:Nim,
    if (InPar.IsSkip(Iim)),
       % skip image - do not define a group
       IndGroup(Iim).Igroup = NaN;
    else
       NewHead = cell_fitshead_getkey(Sim(Iim).(HeaderField),Keys,'NaN');
       if (isempty(Group)),
           Igroup = Igroup + 1;
           Group(Igroup).Vals   = NewHead(:,2);
           IndGroup(Iim).Igroup = Igroup;       
       else
           % check if NewHead(:,2) is identical to one of the existing groups
           Found = false;
           for Ig=1:1:Igroup,
               for Ikey=1:1:Nkeys,
                   Check(Ikey) = strcmp(Group(Ig).Vals{Ikey},NewHead{Ikey,2});
               end
               if (all(Check)),
                   % group found
                   IndGroup(Iim).Igroup = Ig;
                   Found = true;
               end
           end
           if (~Found),
               % open a new group
               Igroup = Igroup + 1;
               Group(Igroup).Vals   = NewHead(:,2);
               IndGroup(Iim).Igroup = Igroup;      
           end
       end
    end
end

          
   
   
       
