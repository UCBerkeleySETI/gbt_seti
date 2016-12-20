function Flag=query_columns(Cat,Col,QueryString)
%--------------------------------------------------------------------------
% query_columns function                                            ImPhot
% Description: Query a table or catalog using a string containing logical
%              operations on column names.
% Input  : - A table, matrix or a structure or SIM. The structure or SIM
%            should contain a 'Cat', 'Col' and 'ColCell' fields.
%          - Optional cell array of column names. By default it will 
%            attempt to get the column names from the table properties
%            or structure Col fields.
%          - Query string. E.g., 'Mag>17 & Mag<18 & Err<0.1' or
%            'Mag+Err';
%            Alternativly this can be  a logical (true or false).
%            In this case will return a vector of the same logical
%            per catalog line.
% Output : - A vector of the query string results.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% BUGS   : BE CAREFULL! If colun name is a substring of another
%          column name this function will not work properly.
% Example: [Flag]=query_columns(rand(10,3),{'RA','Dec','Mag'},'Mag>0.5 & RA>0.5');
%          [Flag]=query_columns(rand(10,3),{'RA','Dec','Mag'},true);
%          [Val]=query_columns(rand(10,3),{'RA','Dec','Mag'},'RA+Dec ');
% Reliable: 2
%--------------------------------------------------------------------------

CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';
    


if (isstruct(Cat) || issim(Cat)),
    if (islogical(QueryString)),
        Nline = size(Cat.(CatField),1);
        QueryString = true(Nline,1) & QueryString;
    else
        ColCell = Cat.(CatColCellField);
        if (istable(Cat.(CatField))),
            BaseReplaceString = sprintf('Cat.%s.%%s',CatField);
        else
            BaseReplaceString = sprintf('Cat.%s(:,Cat.%s.%%s)',CatField,CatColField);
        end
    end
elseif (istable(Cat)),
    if (islogical(QueryString)),
        Nline = size(Cat,1);
        QueryString = true(Nline,1) & QueryString;
    else
        ColCell = Cat.Properties.VariableNames;
        BaseReplaceString = sprintf('Cat.%%s');
    end
else
    if (islogical(QueryString)),
        Nline = size(Cat,1);
        QueryString = true(Nline,1) & QueryString;
    else
        ColCell = Col;
        Col     = cell2struct(num2cell(1:1:length(ColCell)),ColCell,2); % used in the eval command
        BaseReplaceString = sprintf('%s(:,%s.%%s)',CatField,CatColField);
    end                      
end



if (islogical(QueryString)),
    Flag = QueryString;
else
    Ncol = numel(ColCell);
    NewQueryString = QueryString;
    for Icol=1:1:Ncol,
        ReplaceString = sprintf(BaseReplaceString,ColCell{Icol});
        NewQueryString = regexprep(NewQueryString,sprintf('%s',ColCell{Icol}),ReplaceString);
    end
    %sprintf('Flag = %s;',NewQueryString)
    eval(sprintf('Flag = %s;',NewQueryString));
    %sum(Flag)
end