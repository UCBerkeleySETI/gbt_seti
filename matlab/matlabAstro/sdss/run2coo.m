function [F,Object]=run2coo(ID,DR);
%------------------------------------------------------------------------------
% run2coo function                                                        sdss
% Description: Convert SDSS run/rerun/camcol/field/object ID to
%              coordinates.
% Input  : - Matrix of SDSS objects ID: [run rerun camcol field object],
%            where the object is optional.
%          - String containing SDSS Data release, default is 'DR8',
%            or alternatively the SDSS_DR?_Fields_All_PolySort.mat matrix.
% Output : - Structure containing fields coordinates in the following
%            entries:
%            .CornerRA   - Matrix containing 4 columns with the field
%                          corneres J2000.0 RA coordinates.
%            .CornerDec  - Matrix containing 4 columns with the field
%                          corneres J2000.0 Dec coordinates.
%            .MJD        - Matrix containing MJD of [u g r i z] bands.
%            .Type       - Vector of field category:
%                          1: only in Best
%                          2: only in Target
%                          3: both in Target and Best
%            .Ind        - Vector of indices of fields in the 
%                          SDSS_DR?_Fields_All_Poly.mat matrix
%            .IndSort    - Vector of indices of fields in the 
%                          SDSS_DR?_Fields_All_PolySort.mat matrix
%          - Structure containing the object coordinates in the
%            following fields:
%            .RA         - Vector of J2000.0 RA, per object [deg].
%            .Dec        - Vector of J2000.0 Dec, per object [deg].
%            .Type       - Vector of object type.
%            .Mag        - Matrix of modelMag in the [u g r i z] bands.
%            .MagErr     - Matrix of modelMagErr in the [u g r i z] bands.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                     April 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [F,Obj]=run2coo([3813 41 6 60 1]);
% Reliable: 1
%------------------------------------------------------------------------------

DefDR = 'DR8';

if (nargin==1),
   DR = DefDR;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isstr(DR)==1),
   eval(sprintf('Fields=load(''SDSS_%s_Fields_All_PolySort.mat'');',DR));
   FN     = fieldnames(Fields);
   Fields = getfield(Fields,FN{1});
else
   Fields = DR;
   clear DR;
end

Nid = size(ID,1);
F.CornerRA     = zeros(Nid,4);
F.CornerDec    = zeros(Nid,4);
F.MJD          = zeros(Nid,5);
F.Type         = zeros(Nid,1);
F.Ind          = zeros(Nid,1);
F.IndSort      = zeros(Nid,1);
for Iid=1:1:Nid,
   I = find(ID(Iid,1)==Fields(:,1) & ID(Iid,2)==Fields(:,2) & ID(Iid,3)==Fields(:,3) & ID(Iid,4)==Fields(:,4));

   F.CornerRA(Iid,:)   = Fields(I,10:13);
   F.CornerDec(Iid,:)  = Fields(I,14:17);
   F.MJD(Iid,:)        = Fields(I,5:9);  
   F.Type(Iid,:)       = Fields(I,18);
   F.Ind(Iid,:)        = Fields(I,19);
   F.IndSort(Iid,:)    = I;
end


if (nargout>1 & size(ID,2)>4),   
   Object.RA      = zeros(Nid,1);
   Object.Dec     = zeros(Nid,1);
   Object.Type    = zeros(Nid,1);
   Object.Mag     = zeros(Nid,5);
   Object.MagErr  = zeros(Nid,5);
   for Iid=1:1:Nid,
      % query object
      Q{1} = {'ra','dec','type','modelMag_u','modelMag_g','modelMag_r','modelMag_i','modelMag_z','modelMagErr_u','modelMagErr_g','modelMagErr_r','modelMagErr_i','modelMagErr_z'};
      Q{2} = {'PhotoObjAll'};
      Q{3} = sprintf('run=%d and rerun=%d and camcol=%d and field=%d and obj=%d',ID(Iid,:));
      [Cat,Msg] = run_sdss_sql(Q,[],'cell');
      
      Object.RA(Iid)       = Cat(1);
      Object.Dec(Iid)      = Cat(2);
      Object.Type(Iid)     = Cat(3);
      Object.Mag(Iid,:)    = Cat(4:8);
      Object.MagErr(Iid,:) = Cat(9:13);
   end
end
