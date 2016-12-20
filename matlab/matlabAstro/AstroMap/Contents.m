%
% Contents file for package: AstroMap
% Created: 29-Dec-2015
%---------
% amapproj.m :  Plot a list of coordinates (as dots/lines) on a map with a chosen projection.
% plot_monthly_smap.m :  Plot a monthly sky map with a naked eye stars for a given time and observer Geodetic position. Optionaly mark planets position, constellations and the milky way.
% plot_smap.m :  Given a star catalog plot star map with optional magnitude/color/type/proper motion ('/cy).
% pr_aitoff.m :  Project coordinates (longitude and latitude) using equal area Aitoff projection.
% pr_albers.m :  Project coordinates (longitude and latitude) using the Albers Equal-Area projection. The coordinates are projected on ellipse with axis ratio of 2:1.
% pr_azimuthal_equidist.m :  Project longitude and latitude using Azimuthal equidistant projection (constant radial scale).
% pr_bonne.m :  Project coordinates (longitude and latitude) using the Bonne projection.
% pr_cassini.m :  Project coordinates (longitude and latitude) using the Cassini projection.
% pr_conic.m :  Project coordinates (longitude and latitude) using the Conic projection.
% pr_cylindrical.m :  Project coordinates (longitude and latitude) using a general cylindrical projection.
% pr_gnomonic.m :  Project coordinates (longitude and latitude) using the Gnomonic non conformal projection This is a nonconformal projection from a sphere center in which orthodromes are stright lines.
% pr_hammer.m :  Project coordinates (longitude and latitude) using the Hammer projection. The coordinates are projected on an ellipse with axis ratio of 2:1.
% pr_hammer_aitoff.m :  Project coordinates (longitude and latitude) using equal area Hammer-Aitoff projection used in the FITS/WCS standard.
% pr_ignomonic.m :  Project coordinates (X and Y) using the inverse Gnomonic non conformal projection,
% pr_ihammer_aitoff.m :  Project coordinates (longitude and latitude) using the inverse of the equal area Hammer-Aitoff projection used in the FITS/WCS standard.
% pr_mercator.m :  Project coordinates (longitude and latitude) using the Mercator projection.
% pr_mollweide.m :  Project coordinates (longitude and latitude) using the equal area Mollweide projection.
% pr_parabolic.m :  Project coordinates (longitude and latitude) using the Parabolic projection.
% pr_planis.m :  Project longitude and latitude using a 'planisphere projection'.
% pr_polar.m :  Project coordinates (longitude and latitude) using the polar projection (from north pole).
% pr_sinusoidal.m :  Project coordinates (longitude and latitude) using the Sinusoidal projection.
% pr_stereographic.m :  Project coordinates (longitude and latitude) using the Stereographic projection. This is a map projection in which great circles and Loxodromes are logarithmic spirals.
% pr_stereographic_polar.m :  Project coordinates (longitude and latitude) using the Stereographic polar projection. This projection preservs angles.
% pr_xy.m :  Project coordinates (longitude and latitude) to X-Y projection (no transformation).
% prep_dss_fc.m :  Prepare DSS finding charts, with labels, compass, and slits.
% projectcoo.m :  Project coordinates from longitude and latitude to X/Y using a specified projection.
% spherical_tri_area.m :  Given three coordinates on a sphere, calculate the area of a spherical triangle defined by these three points.
% usnob1_map.m :  Plot a finding chart using a local copy of the USNO-B2.0 catalog. The user can select between b/w stars or color stars (with their O-E color index color coded). If O-E is not available then stars are plotted in black. The user can overplot known galaxies from the PGC catalog. In the color option the edge of probable stellar objects is marked by black circle. The user can overplot known bright stars (VT<11) for which spikes and saturation artifact can be seen.
