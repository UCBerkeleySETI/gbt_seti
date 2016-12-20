function NewStruct=structcut(Struct,Ind);
%---------------------------------------------------------------------------
% structcut function                                                General
% Description: Given a structure and a vector of indices, select
%              from each field in the structure only the rows in each
%              field which are specified by the vector of indices.
% Input  : - Structure.
%          - Vector of indices.
% Output : - new structure, with the selected rows only.
% Tested : Matlab 7.3
%     By : Eran O. Ofek                      July 2008
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%---------------------------------------------------------------------------

NewStruct = Struct;
FieldNames = fieldnames(Struct);
N = length(FieldNames);
for I=1:1:N,
   if (isstruct(getfield(Struct,FieldNames{I}))==1),
      % call the function recursively.
      NS = structcut(getfield(Struct,FieldNames{I}),Ind);
      NewStruct=setfield(NewStruct,FieldNames{I},NS);
   else
      FieldCont = getfield(Struct,FieldNames{I});
      FieldCont = FieldCont(Ind,:);

      NewStruct=setfield(NewStruct,FieldNames{I},FieldCont);
   end
end
