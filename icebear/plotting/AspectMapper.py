"""

AspectMapper.py

 This is an aspect angle mapping library which is used to compute aspect
 angle maps for scattering from ionospheric irregularities. Currently it does
 not support raytracing or digital terrain files to allow for accurate
 coverage estimation at low frequencies (below 70 MHz) or complicated physical
 terrains. Antenna patterns are also not incorporated although convolution of an
 antenna pattern with the mapper output should produce reasonable results.

$Id: AspectMapper.py 2477 2010-01-28 15:20:53Z flind $

"""


# import the geomagnetic reference field library
from builtins import object

import icebear.plotting.igrf as igrf
import math

import numpy


# for demo plot

#import mpl_toolkits.basemap
import matplotlib
import matplotlib.cm
import matplotlib.pyplot

# local mapper classes

class Location(object):
    """
        The Location class provides an object to handle geodedic locations.
    """
    def __init__(self, altitude, latitude, longitude):
        # altitude in meters
        # latitude in degrees north
        # longitude in degrees east
        self.altitude = altitude
        self.latitude = latitude
        self.longitude = longitude

    def getAltitude(self):
        return self.altitude

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude

    def getPosition(self):
        return (self.altitude,self.latitude,self.longitude)
    


class AspectAngle(object):
    """
        
        The AspectAngle class is initialized with two locations A and B
        along with latitude and longitude definition of fields of view. A
        limiting aspect angle is also provided. The methods allow computation
        of the magnetic aspect angle using the IGRF magnetic field model and a
        vector direction from and to each location. Regions outside the aspect
        limit or field of view are set to NaN values. 
        
        The 2005 IGRF is currently used although it should be possible to 
        use other coefficient sets through modification of the IGRF loader code.  
         
    """
    def __init__(self, locationA, locationB, fovA, fovB, aspect_limit,igrf_year=2010.0):

        self.A = locationA
        self.B = locationB
        self.fovA = fovA
        self.fovB = fovB
        self.aspect_limit = aspect_limit
        self.igrf_year = igrf_year

        # load the igrf model 2005
        
        self.igrf_model = igrf.IGRF(self.igrf_year)

        # compute the global cartesian coordinates for locations A and B

    def global_cartesian(self, location):
        """
            Converts a given location to global cartesian coordinates. 
        """

        radians = numpy.pi/180.0

        alt = location.altitude
        slat = sin(location.latitude*radians)
        clat = cos(location.latitude*radians)
        elong = location.longitude

        # now convert from geodedic to geocentric
        tmpA = 40680925
        tmpB = 40408585

        tmp_one = tmpA * clat * clat
        tmp_two = tmpB * slat * slat

        tmp_three = tmp_one + tmp_two

        tmp_four = sqrt(tmp_three)

        R = sqrt(alt*(alt + 2.0*tmp_four) + (tmpA*tmp_one + tmpB*tmp_two)/tmp_three)
        cd = (alt + tmp_four)/R
        sd = (tmpA - tmpB)/tmp_four * slat * clat/R

        tmp_slat = slat
        
        slat = slat*cd - clat * sd
        clat = clat*cd - tmp_slat * sd

        # okay now compute the global cartesian coordinates
        
        cphi = cos(elong/radians)
        sphi = sin(elong/radians)

        x = R*clat*cphi
        y = R*clat*sphi
        z = R*slat

        return numpy.array((x,y,z))

    def norm(self, v):
        """ return the L2 norm of the 3 element vector. """
        return numpy.sqrt(numpy.dot(v,v.conj()))

    def llh2xyz(self,loc):
        """ convert from WGS84 to ECEF coordinates"""

        lat = loc.latitude
        long = loc.longitude
        h = loc.altitude
        
        # Convert lat, long, height in WGS84 to ECEF X,Y,Z
        # lat and long given in decimal degrees.
        # altitude should be given in meters

        lat = lat/180.0*numpy.pi       # converting to radians
        long = long/180.0*numpy.pi      # converting to radians
        a = 6378137.0;                  # earth semimajor axis in meters 
        f = 1/298.257223563             # reciprocal flattening 
        e2 = 2*f-numpy.power(f,2)       # eccentricity squared 
 
        chi = numpy.sqrt(1-e2*numpy.power((numpy.sin(lat)),2)); 
        X = (a/chi +h)*numpy.cos(lat)*numpy.cos(long);
        Y = (a/chi +h)*numpy.cos(lat)*numpy.sin(long);
        Z = (a*(1-e2)/chi + h)*numpy.sin(lat);

        return numpy.array([X,Y,Z])

    def aspect_angle(self, locT):
        """
            For the provided location compute the magnetic aspect angle for vectors
            from location A to location B that intersect location T. 
        """

        # compute the global cartesian coordinate location for the target point T
        #
        
        cartA = self.llh2xyz(self.A)
        cartB = self.llh2xyz(self.B)
        cartT = self.llh2xyz(locT)

        # compute the difference vectors between the target T and each location A and B

        uvA = cartT - cartA
        uvB = cartT - cartB

        # compute the distance (norm) of the vectors

        dA = self.norm(uvA)
        dB = self.norm(uvB)

        # check FOV and if it is okay then compute the aspect angle
        #print "dA", dA/1000.0, "dB", dB/1000.0
        
        if (dA <= self.fovA) and (dB <= self.fovB):

            # compute the magnetic field at the points A, B, T
            #bfieldA = self.igrf_model.B(self.A.altitude/1000.0, self.A.latitude, self.A.longitude)
            #bfieldB = self.igrf_model.B(self.B.altitude/1000.0, self.B.latitude, self.B.longitude)
            
            bfieldT = self.igrf_model.B(locT.altitude/1000.0, locT.latitude, locT.longitude)

            # grab the global cartesian B field for vector manipulations
            # BcartA = numpy.array(bfieldA[2])
            # BcartB = numpy.array(bfieldB[2])
            BcartT = numpy.array(bfieldT[2])


            # make unit vectors
            uvA = uvA/dA
            uvB = uvB/dB

            bisector = uvA + uvB
            bisector = bisector / self.norm(bisector)

            # make B field unit vector
            BcartT_uv = BcartT/self.norm(BcartT)

            # compute aspect angle
            aspect_angle = 90-180.0*(numpy.arccos(numpy.dot(BcartT_uv,bisector)))/numpy.pi

            if numpy.abs(aspect_angle) > self.aspect_limit:
                aspect_angle = numpy.NaN
        
        else:
            # numpy NaN
            aspect_angle = numpy.NaN

        return aspect_angle

class AspectMapper(object):
    """
        The AspectMapper class takes a set of TX and RX locations, a latitude / longitude
        range (arrays), a computation altitude, a field of view limit, and a aspect angle limit. 
        It computes a map that consists of an 2-D array of magnetic aspect angle as a function
        of latitude and longitude. The produced map is the the minimum aspect angle from all 
        TX and RX pairs through the volume. The TX and RX location may be the same for monostatic
        radar calculations or different for multistatic computations.   
    """
    def __init__(self, tx_locations, rx_locations, latitude_range, longitude_range, altitude, fov_limit, aspect_limit, igrf_year):

        self.tx_locations = tx_locations
        self.rx_locations = rx_locations
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.altitude = altitude
        self.aspect_limit = aspect_limit
        self.fov_limit = fov_limit
        self.igrf_year = igrf_year

        # create empty map
        self.map = None
        self.clearMap()

    def clearMap(self):
        # default is NaN
        self.map = numpy.ones((len(self.latitude_range),len(self.longitude_range)))*numpy.NaN

    def getMap(self):
        return self.map        

    def computeMap(self):

        for tx in self.tx_locations:
            for rx in self.rx_locations:
                # create a TX RX aspect angle pair
                aa_pair = AspectAngle(tx,rx,self.fov_limit,self.fov_limit,self.aspect_limit,self.igrf_year)

                lon_index = 0

                for lon in self.longitude_range:

                    lat_index = 0

                    for lat in self.latitude_range:
                        loc = Location(self.altitude, lat, lon)
                        # compute the aspect angle for this target location
                        aspect = aa_pair.aspect_angle(loc)
                        if numpy.isnan(self.map[lat_index,lon_index]):
                            self.map[lat_index,lon_index] = aspect
                        elif numpy.abs(aspect) < numpy.abs(self.map[lat_index,lon_index]):
                            self.map[lat_index,lon_index] = aspect
                        #if not numpy.isnan(self.map[lat_index,lon_index]):
                        #    print self.map[lat_index,lon_index]
                        lat_index += 1

                    lon_index +=1


#if __name__ == '__main__':

    """
        The example program is just for a basic example. Use these classes separately
        for particular applications. Here we plot a monostatic radar aspect angle map
        for the Millstone Hill radar site. 
    """
"""    
    # Millstone Hill MISA
    misa = Location(146.0,42.61950,288.50827)
    
    # monostatic map
    tx = [misa]
    rx = [misa]

    lat_range = numpy.arange(35,55,0.5,'f')
    lon_range = numpy.arange(275,300,0.5,'f')

    altitude = 110.0*1.0E3    # in meters
    fov_limit = 1100.0*1.0E3  # in meters
    aspect_limit = 2.0
    
    AM = AspectMapper(tx,rx,lat_range,lon_range,altitude,fov_limit,aspect_limit)

    AM.computeMap()

    map = AM.getMap()

    mm = numpy.ma.masked_where(numpy.isnan(map), map)
    
    masked_map = numpy.abs(mm) * 10.0

    # setup figure
    fig = matplotlib.pyplot.figure()

    # setup basemap
    print numpy.min(lon_range), numpy.min(lat_range), numpy.max(lon_range), numpy.max(lat_range)
    m = mpl_toolkits.basemap.Basemap(resolution='i',projection='merc',
                                     llcrnrlon = numpy.min(lon_range),
                                     llcrnrlat = numpy.min(lat_range),
                                     urcrnrlon = numpy.max(lon_range),
                                     urcrnrlat = numpy.max(lat_range))
    ax = fig.add_axes([0.1,0.1,0.7,0.7])

    lons,lats = numpy.meshgrid(lon_range,lat_range)
    x,y = m(lons,lats)
    m.contour(x,y,masked_map,int(20.0/aspect_limit),linewidths=0.5,colors='k')
    m.contourf(x,y,masked_map,int(40.0/aspect_limit),cmap=matplotlib.cm.hsv)

    pos = ax.get_position()
    l, b, w, h = pos.bounds
    cax = matplotlib.pyplot.axes([l+w+0.075, b, 0.05, h])
    
    matplotlib.pyplot.colorbar(drawedges=True, cax=cax) # draw colorbar
    matplotlib.pyplot.axes(ax) 
    # draw coastlines and political boundaries.
    m.drawcoastlines()
    m.drawmapboundary()
    # draw parallels and meridians.
    parallels = numpy.arange(-80.,90,5.)
    m.drawparallels(parallels,labels=[1,0,0,1])
    meridians = numpy.arange(0.,360.,5.)
    m.drawmeridians(meridians,labels=[1,0,0,1])
    #matplotlib.pyplot.xlabel('Longitude')
    #matplotlib.pyplot.ylabel('Latitude')

    matplotlib.pyplot.title('Aspect Angle Attenuation')
    #print 'plotting...'
    matplotlib.pyplot.show()

    
                
"""
        
