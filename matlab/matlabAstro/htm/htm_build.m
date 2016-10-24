function [HTM,LevList]=htm_build(Level)
%------------------------------------------------------------------------------
% htm_build function                                                       htm
% Description: Build Hierarchical Triangular Mesh (HTM) structure.
%              This structure can be use for fast searches of data
%              in catalogs on a sphere.
% Input  : - The number of levels in the HTM structure.
% Output : - The HTM structure array with the follwoing fields.
%            .level  - Level depth  index (0 for the first level).
%            .coo    - Coordinates of triangular mesh [Long, Lat] in
%                      radians. (3x2 matrix).
%                      The coordinates are ordered such that the
%                      right-hand rule is pointing toward the
%                      center of the polygon.
%            .cosd   - Cosine directions of triangular mesh.
%                      3x3 matrix in which each line corresponds to
%                      each vertex of the triangle.
%            .id     - Triangle id. This is a vector in which the
%                      number of elements equal to the number of levels.
%                      The first level is between 0 to 7 and all
%                      the other levels are between 0 to 3.
%            .father - The index of the father.
%                      If empty matrix then there is no father.
%            .son    - Vector of indices of all sons.
%                      If empty matrix then there are no sons.
%          - A structure array of list of levels (LevList).
%            Number of elements corresponds to the number of levels.
%            The structure contains the following fields:
%            .level - Level depth  index (0 for the first level).
%            .ptr   - Vector of indices of all elements in HTM
%                     which are in this level.
%            .side  - Length of side of triangles in level [radians].
% Tested : Matlab 7.11
%     By : Eran O. Ofek                      July 2011
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [HTM,LevList]=htm_build(4);
% Reliable: 2
%------------------------------------------------------------------------------


Ind = 0;

%Nel = 8;
%for Il=1:1:Level-1,
%  Nel = Nel + Nel.*4;
%end
%TM = struct('level',cell(1,Nel),...
%            'coo',cell(1,Nel),...
%            'cosd',cell(1,Nel),...
%            'id',cell(1,Nel),...
%            'father',cell(1,Nel),...
%            'son',cell(1,Nel),...
%            'cat',cell(1,Nel));

% build northern zero-level HTM
for I=1:1:4,
   Ind = Ind + 1;
   HTM(Ind).level    = 0;
   HTM(Ind).coo      = [0 0; pi./2 0; 0 pi./2];
   HTM(Ind).coo(:,1) = HTM(Ind).coo(:,1) + pi./2.*(I-1);
   [CD1, CD2, CD3]   = coo2cosined(HTM(Ind).coo(:,1),HTM(Ind).coo(:,2));
   HTM(Ind).cosd     = [CD1, CD2, CD3];  
   HTM(Ind).id       = Ind-1;
   HTM(Ind).father   = [];
   %HTM(Ind).brother  = [1:1:8];
   HTM(Ind).son      = [];
   HTM(Ind).cat      = [];
end


% build south hemisphere zero-level HTM
for I=1:1:4,
   Ind = Ind + 1;
   HTM(Ind).level    = 0;
   HTM(Ind).coo      = [0 0; 0 -pi./2; pi./2 0];
   HTM(Ind).coo(:,1) = HTM(Ind).coo(:,1) + pi./2.*(I-1);
   [CD1, CD2, CD3]   = coo2cosined(HTM(Ind).coo(:,1),HTM(Ind).coo(:,2));
   HTM(Ind).cosd     = [CD1, CD2, CD3];  
   HTM(Ind).id       = Ind-1;
   HTM(Ind).father   = [];
   %HTM(Ind).brother  = [1:1:8];
   HTM(Ind).son      = [];
   HTM(Ind).cat      = [];
   HTM(Ind).cat      = [];
end

LevList(1).level = 0;
LevList(1).ptr   = [1:1:8];
LevList(1).side  = pi./2;


if (Level==0),
   % stop
else
   % build HTM recursively
   [HTM,LevList] = htm_build_son(HTM,LevList,Level);
end
 
