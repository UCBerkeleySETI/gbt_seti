%
% Contents file for package: ephem
% Created: 29-Dec-2015
%---------
% abberation.m :  Calculate the position of a star corrected for abberation deflection due to the Sun gravitational field. Rigoursly, these is applied after accounting for the light travel time effect and light deflection and before the precession and nutation are being applied.
% add_offset.m :  Add an offset specified by angular distance and position angle to celestial coordinates.
% airmass.m :  Given the JD, object celestial coordinates, and observer Geodetic coordinates, calculating the airmass of the object.
% alt2ha.m :  Given an object altitude and declination and the observer latitude, return the corresponding Hour Angle.
% altha2dec.m :  Given Altitude and Hour Angle of an object and the observer latitude, calculate the object Declination. There may be up to two solutions for the Declination.
% angle_in2pi.m :  Convert an angle to the range 0 to 2.*pi.
% apsides_precession.m :  First order estimation of the GR precession of the line of apsides.
% area_sphere_polygon.m :  Calculate the area of a polygon on a sphere, where the polygon sides are assumed to be great circles. If the polygon is not closed (i.e., the first point is identical to the last point) then, the program close the polygon.
% asteroid_magnitude.m :  Calculate the magnitude of minor planets in the HG system. Valid for phase angles (Beta) in range 0 to 120 deg.
% astrometric_binary.m :  Given orbital elements of an astrometric binary, predicts its sky position as a function of time.
% calc_vsop87.m :  Calculate planetary coordinates using the VSOP87 theory.
% cel_annulus_area.m :  Calculate the area within a celestial annulus defined by an inner radius and outer radius of two concentric small circles.
% celestial_circ.m :  Calculate grid of longitude and latitude of a small circle on the celestial sphere.
% coco.m :  General coordinate convertor. Convert/precess coordinate from/to Equatorial/galactic/Ecliptic coordinates.
% constellation.m :  Find the constellations in which celestial coordinates are located.
% convert_year.m :  Convert between different types of years. For example, this program can convert Julian years to Besselian years or JD and visa versa.
% convertdms.m :  Convert between various representations of coordinates and time as sexagesimal coordinates, degrees and radians.
% coo2cosined.m :  Convert coordinates to cosine directions in the same reference frame. See also: cosined.m, cosined2coo.m
% cosined.m :  Cosine direction transformation. Convert longitude and latitude to cosine direction and visa versa. See also: coo2cosined.m, cosined2coo.m
% cosined2coo.m :  Convert cosine directions to coordinates in the same reference frame. See also: cosined.m, coo2cosined.m
% daily_observability.m :  Plot the observability of a given object from a give location on Earth during one night. This program will plot the object Alt during the night, along with the Sun/Moon alt and the excess in sky brightness due to the Moon.
% days_in_month.m :  Calculate the number of days in a given Gregorian or Julian month.
% delta_t.m :  Return \Delta{T} at a vector of Julian days. DeltaT is defined as ET-UT prior to 1984, and TT-UT1 after 1984 (= 32.184+(TAI-UTC)-(UT1-UTC)).
% dnu_dt.m :  Calculate dnu/dt and dr/dt for elliptical orbit using the Kepler Equation.
% earth_gravity_field.m :  Calculate the Earth gravity field for a set of locations. For both rotating and non rotating Earth. Mean IRTF pole is assumed.
% earth_vel_ron_vondrak.m :  Calculate the Earth barycentric velocity in respect to the mean equator and equinox of J2000.0, using a version of the Ron & Vondrak (1986) trigonometric series.
% easter_date.m :  Calculate the date of Easter for any Gregorian year.
% eccentric2true_anomaly.m :  Convert Eccentric anomaly to true anomaly.
% ecliptic2helioecliptic.m :  Transform ecliptic longitude to Helio-ecliptic longitude.
% elements_1950to2000.m :  Convert solar system orbital elements given in the B1950.0 FK4 reference frame to the J2000.0 FK5 reference frame.
% equinox_solstice.m :  Calculate the approximate time of Equinox and Solstice for a given list of years. Accurate to about 100s between year -1000 to 3000.
% gauss_grav_const.m :  Get the analog of the Gaussian gravitational constant for a system with a given primary mass, secondary mass and unit distance. This program is useful in order to apply kepler.m for non-solar system cases.
% geoc2geod.m :  Convert Geocentric coordinates to Geodetic coordinates using specified reference ellipsoid.
% geocentric2lsr.m :  Approximate conversion of geocentric or heliocentric velocity to velocity relative to the local standard of rest (LSR).
% geod2geoc.m :  Convert Geodetic coordinates to Geocentric coordinates using specified reference ellipsoid.
% get_atime.m :  Get current time, date, JD and LST.
% get_moon.m :  Get Moon position (low accuracy).
% get_sun.m :  Get Sun position (low accuracy).
% ha2alt.m :  Given Hour Angle as measured from the meridian, the source declination and the observer Geodetic latitude, calculate the source altitude above the horizon and its airmass.
% ha2az.m :  Given Hour Angle as measured from the meridian, the source declination and the observer Geodetic latitude, calculate the horizonal source azimuth
% hardie.m :  Calculate airmass using the Hardie formula.
% hardie_inv.m :  Inverse Hardie airmass function. Convert airmass to zenith distance.
% horiz_coo.m :  Convert Right Ascension and Declination to horizontal coordinates or visa versa.
% inside_celestial_box.m :  Given a list of celestial coordinates, and a box center, width and height, where the box sides are parallel to the coorinate system, check if the coordinates are within the box.
% interp_coo.m :  Interpolate on celestial ccordinates as a function of time. Use the built in matlab interpolation functions.
% jd2date.m :  Convert Julian days to Gregorian/Julian date.
% jd2mjd.m :  Convert JD to MJD
% jd2year.m :  Convert Julian day to Julian or Besselian years.
% julday.m :  Convert Julian/Gregorian date to Julian Day.
% julday1.m :  Convert Gregorian date in the range 1901 to 2099 to Julian days (see also: julday.m).
% jup_meridian.m :  Low accuracy formula for Jupiter central meridian.
% jup_satcurve.m :  Plot monthly curves of the relative position of the Galilean satellites of Jupiter.
% jupiter_map.m :  Plot Jupiter image as observed from Earth at a given time.
% keck_obs_limits.m :  Given a date and object celestial positions, calculate the rise and set time of an object in Keck observatory, given the Nasmyth mount limits.
% kepler3law.m :  Calculate the velocity, semi-major axis and period of a system using the Kepler third law.
% kepler_elliptic.m :  Solve Kepler equation (M = E - e sin E) and find the true anomaly and radius vector for elliptical orbit (i.e., 0<=e<1). The function requires the time since periastron (t-T), the periastron distance (q) and the Gaussian constant for the system and units (k; see gauss_grav_const.m).
% kepler_elliptic_fast.m :  Fast solver for the Kepler equation for elliptic orbit. This is a simpler version of kepler_elliptic.m
% kepler_hyperbolic.m :  Solve Kepler equation (M = e sinh H - H) and find the true anomaly and radius vector for hyperbolic orbit (i.e., e>1). The function requires the time since periastron (t-T), the periastron distance (q) and the Gaussian constant for the system and units (k; see gauss_grav_const.m).
% kepler_lowecc.m :  Solve the Kepler Equation for low eccentricity using a series approximation. This is typically better than 1" for e<0.1.
% kepler_parabolic.m :  Solve Kepler equation (M = E - e sin E) and find the true anomaly (related to the eccentric anomaly, E) and radius vector for parabolic orbit (i.e., e=1). The function requires the time since periastron (t-T), the periastron distance (q) and the Gaussian constant for the system and units (k; see gauss_grav_const.m).
% lst.m :  Local Sidereal Time, (mean or apparent), for vector of JDs and a given East Longitude.
% meteor_multistation.m :  Given a list of observers geodetic coordinates in which the first point is a reference point; the azimuth and altitude in which an observer (located in the reference position) is looking to; and the height (H) of a meteor trail - calculate the azimuth and altitude in which observers in other points should look to, in order to detect the same meteor. The function takes into acount the Earth curvature (first order).
% meteors_db.m :  Return a meteor showers database (not complete).
% mjd2jd.m :  Convert MJD to JD
% month_name.m :  Given a month number return a string with a month name.
% moon_elp82.m :  Calculate accurate ELP2000-82 ecliptic coordinates of the Moon, referred to the inertial mean ecliptic and equinox of date. This function was previously called moonpos.m.
% moon_illum.m :  Low accuracy Moon illuminated fraction
% moon_phases.m :  Return a list of moon phases in range of dates.
% moon_sky_brightness.m :  Given the date, object equatorial coordinates, and observer geodetic position, calculate the excess in sky brightness (V-band) in the object celestial position. The function utilize the algorithm by Krisciunas & Schaefer (1991).
% moon_sky_brightness_h.m :  Given the horizontal coordinates of the Moon and an object and the observer geodetic position, calculate the excess in sky brightness (V-band) in the object celestial position. The function utilize the algorithm by Krisciunas & Schaefer (1991).
% mooncool.m :  Calculate low-accuracy topocentric equatorial coordinates of the Moon, referred to the equinox of date.
% moonecool.m :  Calculate low accuracy geocentric ecliptical coordinates of the Moon, referred to the mean equinox of date. Accuracy: in longitude and latitude ~1', distance ~50km To get apparent longitude add nutation in longitude.
% moonlight.m :  Calculate the Moon illumination in Lux on horizontal surface as a function of the Moon altitude, horizontal parallax and Elongation.
% nearest_coo.m :  Given a list of coordinates (with arbitrary number of dimensions), search for the coordinate in list which is the nearest to a given (single) coordinate.
% nutation.m :  Calculate the Nutation in longitude and latitude, and the nutation rotation matrix. This is a low accuracy version based on the IAU 1984 nutation series. See also: nutation1984.m
% nutation1980.m :  Calculate the IAU 1980 Nutation series for a set of JDs.
% nutation2rotmat.m :  Given nutation in longitude and obliquity (in radians) and JD, return the Nutation rotation matrix.
% nutation_lowacc.m :  Low accuracy (~1") calculation of the nutation.
% obliquity.m :  Calculate the obliquity of ecliptic, with respect to the mean equator of date, for a given julian day.
% observatory_coo.m :  Return geodetic coordinates of an observatory.
% obspl.m :  GUI Observation Planer. Plot Alt/Airmass and moon sky brightness as a function of time in night, or yearly visibility plot per celestial object.
% parallactic2ha.m :  Convert parallactic angle, declination and latitude to hour angle. Note that there are two solutions, and the function will return both.
% parallactic_angle.m :  Calculate the parallactic angle of an object. The parallactic is defined as the angle between the local zenith, the object and the celestial north pole measured westwerd (e.g., negative before, and positive after the passage through the southern meridian).
% planar_sundial.m :  Calculate and plot a planar sundial.
% planet_radius.m :  Get planetary radius and flattening factor, and calculate planet angular diameter.
% planets_lunar_occultations.m :  Calculate local circumstences for lunar occultations of Planets and asteroids. Only events in which the planet is above the local horizon will be selected.
% planets_magnitude.m :  Calculate the planets apparent magnitude.
% planets_rotation.m :  Return Planet north pole, rotation rate and the primery meridian.
% ple_earth.m :  Low accuracy planetray ephemeris for Earth. Calculate Earth heliocentric longitude, latitude and radius vector referred to the mean ecliptic and equinox of date. Accuarcy: Better than 1' in long/lat, ~0.001 au in dist.
% ple_jupiter.m :  Low accuracy planetray ephemeris for Jupiter. Calculate Jupiter heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: ~1' in long/lat, ~0.001 au in dist.
% ple_mars.m :  Low accuracy planetray ephemeris for Mars. Calculate Mars heliocentric longitude latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: Better than 1' in long/lat, ~0.001 au in dist.
% ple_mercury.m :  Low accuracy ephemerides for Mercury. Calculate Mercury heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: better than 1' in long/lat, ~0.001 au in dist.
% ple_neptune.m :  Low accuracy planetray ephemeris for Neptune. Calculate Neptune heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: Better than 1' in long/lat, ~0.001 au in dist.
% ple_planet.m :  Low accuracy ephemeris for the main planets. Given a planet name calculate its heliocentric coordinates referred to mean ecliptic and equinox of date. Accuarcy: Better ~1' in long/lat, ~0.001 au in dist.
% ple_saturn.m :  Low accuracy planetray ephemeris for Saturn. Calculate Saturn heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: ~1' in long/lat, ~0.001 au in dist.
% ple_uranus.m :  Low accuracy ephemeris for Uranus. Calculate Uranus heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: ~1' in long/lat, ~0.001 au in dist.
% ple_venus.m :  Planetry Low accuracy ephemeris for Venus. Calculate Venus heliocentric longitude, latitude and radius vector referred to mean ecliptic and equinox of date. Accuarcy: Better than 1' in long/lat, ~0.001 au in dist.
% pm2space_motion.m :  Convert proper motion, radial velocity and parralax to space motion vector in the equatorial system.
% pm_vector.m :  Return the space motion vector given proper motion, parallax and radial velocity. 
% pole_from2points.m :  Given two points on the celestial sphere (in any system) describing the equator of a coordinate system, find one of the poles of this coordinate system.
% proper_motion.m :  Applay proper motion to a acatalog
% refellipsoid.m :  Return data for a given reference ellipsoid of Earth.
% refraction.m :  Estimate atmospheric refraction, in visible light.
% refraction_coocor.m :  Calculate the correction in equatorial coordinates due to atmospheric refraction.
% refraction_wave.m :  Calculate the wavelength-dependent atmospheric refraction and index of refraction based on Cox (1999) formula.
% rise_set.m :  Given an object coordinates and observer position, calculate rise/set/transit times, and azimuth and altitude. The times are in the UT1 (not UTC) system.
% rotm_coo.m :  Generate a rotation matrix for coordinate conversion and precession.
% saturn_rings.m :  Calculate the orientation angles for Saturn's rings.
% search_conj.m :  Search for conjunctions on the celestial sphere between two moving objects given thier coordinates as a function of time.
% search_conj_sm.m :  Search for conjunctions on the celestial sphere between a list of stationary points and a moving object given the coordinates of the moving object as a function of time.
% sky_area_above_am.m :  Calculate sky area observable during the night above a a specific airmass, and assuming each field is observable for at least TimeVis hours.
% skylight.m :  Calculate the total sky illumination due to the Sun, Moon, stars and air-glow, in Lux on horizontal surface as a function of time.
% sphere_dist.m :  Calculate the angular distance and position angle between two points on the celestial sphere.
% sphere_dist_cosd.m :  Calculate the angular distance between a set of two cosine vector directions. This should be used instead of sphere_dist_fast.m only if you have the cosine vectors.
% sphere_dist_fast.m :  Calculate the angular distance between two points on the celestial sphere. See sphere_dist.m (and built in distance.m) for a more general function. This function is ~10 time faster than sphere_dist.m, but it works only with radians and calculate only the distance.
% sphere_offset.m :  Calculate the offset needed to move from a point on the celesial sphere to a second point on the celestial sphere, along longitide (small circle) and latitude (great circle). The needed offsets depends in which axis the offset is done first (longitude or latitude - 'rd' and 'dr' options, respectively).
% sun_rise_set.m :  Given the coordinates and observer position, calculate rise/set/transit/twilight times and azimuth and altitude for the Sun. The accuracy depends on the function used for calculating the solar position. With the default sun-position function, the geometric accuracy is about a few seconds.
% suncoo.m :  Calculate the Sun equatorial coordinates using low accuracy formale. Accuracy : 0.01 deg. in long.
% suncoo1.m :  Calculate the Sun equatorial coordinates using low accuracy formaulae for the range 1950 to 2050. Accuracy : 0.01 deg. in long, 0.1m in Equation of Time
% sunlight.m :  Calculate the Sun illumination in Lux on horizontal surface as a function as its altitude in radians.
% tai_utc.m :  Return the TAI-UTC time difference (leap second) for a vector of Julian days. Also return TT-UTC.
% tdb_tdt.m :  Calculate approximate difference between TDT and TDB time scales.
% thiele_innes.m :  Calculate the Thiele-Innes orbital elements.
% thiele_innes2el.m :  Calculate the orbital elements from the Thiele-Innes orbital elements.
% true2eccentric_anomaly.m :  
% trueanom2pos.m :  Given an object true anomaly, radius vector, time and orbital elements and time, calculate its orbital position in respect to the orbital elements reference frame.
% trueanom2vel.m :  Given an object true anomaly, radius vector, their derivatives and orbital elements and time, calculate its orbital position and velocity in respect to the orbital elements reference frame.
% ut1_utc.m :  Return UT1-UTC (also known as DUT1).
% vb_ephem.m :  Given orbital elements of a visual binary, calculate its ephemeris in a give dates.
% wget_eop.m :  Get the table of historical and predicted Earth orientation parameters (EOP) from the IERS web site.
% wget_tai_utc.m :  Get the table of historical TAI-UTC time differences (leap second) from the IERS web site.
% year2jd.m :  Return the Julian day at Jan 1 st of a given list of years.
% yearly_observability.m :  Plot a yearly observability chart for an object.
