'''

    Module igrf.py
    
    $Id: igrf.py 2473 2010-01-28 14:48:49Z flind $

    A python implementation of the magfduw.m Matlab model of IGRF, in particular
    the magfduw pure geodetic coordinate version.

    author : John D. Sahr jdsahr@u.washington.edu
    
typical usage:

python
> from igrf import *
> a = IGRF(2001.5)
> b = a.B(100, 60, 240)

at this point, b will contain four items:
  b[0] = (geodetic X, Y, Z) components
  b[1] = (spherical r, theta, phi) components
  b[2] = (cartesian x, y, z) components
  b[3] = total field strength

all are expressed in nanotesla

%  MAGFDUW
%  Function to compute Earths magnetic field
%  and components: X,Y,Z,T for a given latitude
%  and longitude, date and altitude.
%
%  Usage: out=magfduw(DATE, ALT, COLAT, ELONG);
%
%  DATE  = date of survey            (decimal years)
%  ALT   = altitude of survey relative to sealevel (km +ve up)
%  NLAT  = north latitude            (decimal degrees)
%  ELONG = east longitude of survey  (decimal degrees)
%
% % old output format of magfd() routine
% % Output array out contains components X,Y,Z,T in nanoteslas
% %  T total field in nT
%
% OUT(1:10) scaled in nanoteslas
%
% OUT(1:3) B in local X,Y,Z coordinates (unchanged from original)
% OUT(4:6) B in R, Theta, Phi spherical coordinates
%        note that Theta is "co-latitude" not azimuthal
% OUT(7:9) B in cartesian X, Y, Z coordinates,
%        Z = towards pole star
%        X = equator, through Greenwich meridian
%        y = equator, 90E of Greenwich meridian
% OUT(10) B total magnitude
%
%  ref: IAGA, Division V, Working Group 8, 
%   International geomagnetic reference field, 1995 
%   revision, Geophys. J. Int, 125, 318-321, 1996.
% IGRF2000
% IAGA Working Group V-8 (Chair: Mioara Mandea, Institut de 
% Physique du Globe de Paris, 4 Place Jussieu, 75252 Paris, 
% France. Fax:33 238 339 504, email: mioara@ipgp.jussieu.fr).
% http://www.dsri.dk/Oersted/Field_models
%
% Maurice A. Tivey March 1997
% Mod Dec 1999 (add igrf2000 and y2k compliance
% Uses MATLAB MAT files sh1900.mat to sh2000.mat in 5 yr
% intervals.
% http://deeptow.whoi.edu/matlab.html
%
% modified from magfd.m by John Sahr jdsahr@u.washington.edu
% 13 sept 2004
%
% (*) get rid of ITYPE (always want geodetic coordinates)
% (*) colat -> lat  (This was just confusing)
%
% I have removed some FORTRAN-isms, and gotten rid of some superfluous
% variables which just make it harder to understand.
%
% compare to magfd(year, 1, alt, 90-nlat, elong)
%
'''
import math
import numpy

default_coefficient_file = 'igrf12coeffs.txt'
default_year = 2015.0

class IGRF:
    def __init__(self,
                 year=default_year,
                 coefficient_file=default_coefficient_file):
        '''the constructor: __init__([year],[coeff file])
        will open the IGRF legendre model and set it to the proper
        year, or raise an exception if it can't.'''

        ''' First try to open the coefficient file, and parse it all in
            The format is that provided by NOAA in their IGRF text format.
        '''
        try:
            f = open(coefficient_file)
        except:
            raise IOError('failed to open %s' % coefficient_file)

        self.data = f.readlines()
        self.model_order = numpy.sqrt(len(self.data))
        
        for k in range(len(self.data)):
            line = self.data[k].split()

            # exclude the secular variation value, no projection forward in time.
            for g in range(3, len(line)-1):
                try:
                    line[g] = float(line[g])
                except:
                    raise ValueError('problem reading line %d at element %d' % (k,g))

            self.data[k] = line

        self.first_year = self.data[0][3]
        self.last_year = self.data[0][ - 2]
        
        self.load_gh(year)

    def load_gh(self, year):
        '''do linear interpolation to set the legendre coefficients
        for the indicated year.'''
        if (year <= self.first_year) or (year > self.last_year):
            raise ValueError('year %d out of range %d to %d' % (year, self.first_year, self.last_year))

        k = 3  # the year is in the first line, starting at [3:]
        
        while year > self.data[0][k]:
            k = k + 1

        before = k - 1
        after = k

        a = 1.0 - (year - self.data[0][before]) / 5.0
        b = 1.0 - a

        self.dgh = numpy.array([0])
        for g in self.data[1:]:
            self.dgh = numpy.append(self.dgh,(a * g[before] + b * g[after]))


    def B(self, altitude, north_latitude, east_longitude):
        '''evaluate the IGRF magnetic field model at the desired
        altitude, latitude, and longitude for the (previously
        indicated) decimal year.  The result is specified in several
        ways:
          geodetic  X,Y,Z (north, east, down)
          spherical r, theta, phi (up, south, east)
          cartesian CX, CY, CZ (greenwich long, 90E long, north pole)
        and
          total field strength.'''
        
        rads = numpy.pi / 180.0  # factor for conversion from degrees to radians

        # sin and cos of latitude
        SLAT = numpy.sin(north_latitude * rads)
        CLAT = numpy.cos(north_latitude * rads);

        # sin and cos of longitude, as well as of M*longitude ...
        CL = numpy.array([0, numpy.cos(east_longitude * rads)])
        SL = numpy.array([0, numpy.sin(east_longitude * rads)])
      
        # CONVERSION FROM GEODETIC TO GEOCENTRIC COORDINATES
        
        A2 = 40680925.0;
        B2 = 40408585.0;

        MEAN_EARTH_RADIUS = 6371.2; # in km

        # ONE, TWO, THREE, FOUR are temp variables
    
        ONE = A2 * CLAT * CLAT
        TWO = B2 * SLAT * SLAT
        THREE = ONE + TWO
        FOUR = numpy.sqrt(THREE)
        
        R = numpy.sqrt(altitude * (altitude + 2.0 * FOUR) + (A2 * ONE + B2 * TWO) / THREE)

        # 'D' is some sort of small correction to the
        # polar angle theta, with CD being "cos D" and SD being "sin D"

        CD = (altitude + FOUR) / R
        SD = (A2 - B2) / FOUR * SLAT * CLAT / R

        # correction from from geodetic to spherical latitude
        
        ONE = SLAT
        SLAT = SLAT * CD - CLAT * SD
        CLAT = CLAT * CD + ONE * SD
    
        RATIO = MEAN_EARTH_RADIUS / R

        #  at this point in the program
        #    R = absolute radius in km
        #    SLAT = sin(correct (spherical) lat) = cos(correct theta)
        #    CLAT = cos(correct (spherical) lat) = sin(correct theta)
        #
        #  COMPUTATION OF SCHMIDT QUASI-NORMAL COEFFICIENTS  P AND X(=Q)
        #
        #  initialization of the recursion ...

        P = numpy.array([0,
            2.0 * SLAT,
            2.0 * CLAT,
            4.5 * SLAT * SLAT - 1.5,
            math.sqrt(27.0) * CLAT * SLAT])

        Q = numpy.array([0,
            - CLAT,
            SLAT,
            - 3.0 * CLAT * SLAT,
            math.sqrt(3.0) * (SLAT * SLAT - CLAT * CLAT)])

        # The following loop contains the legendre recursion and
        # summation.  The original magfd.m code had a "for k=1:44"
        # which was a little mysterious, but meant that the loop
        # precisely chewed up the coefficients through 8th order.
        #
        # The following code preserves the same loop structure, but
        # rather than having the loop terminate after a specified
        # number of passes, it terminates on the order N.  This way
        # you can very easily truncate the model if you wish.
        #
        #
        # There are two things going on in the loops:
        #  1. a recursion to generate the Schmidt norm'ed Legendre coefficients
        #  2. the accumulation of the gradient of the magnetic potential
    
        L = 1   # an index which counts through the coefficient in the
                # g/h tables
        M = 1   # the "minor" order of the Legendre Function coefficients
        N = 0   # the "major" order of the Legendre Function coefficients

        K = 0   # auxiliary loop counter
        
        X = 0.0; # accumulates the northward (-theta hat) component of B
        Y = 0.0; # accumulates the eastward (phi hat) component of B
        Z = 0.0; # accumulates the downward (-r hat) component of B

        while N < 12:
            if (N < M):  # the outer loop increments m until it 
                M = 0   # it exceeds n, then resets m, increments n
	        N = N + 1

	        RR = RATIO ** (N + 2)
            # end

            if N > 10:   # termination of the loop at the specified order
                break
            
            K = K + 1    # increment the loop counter explicitly

            if (K > 4):  # need to do recursion for P, Q, SL, CL
                if (M == N):  
                    ONE = numpy.sqrt(1.0 - 0.5 / M)
                    J = K - N - 1;
                        
                    P = numpy.append(P,((1.0 + 1.0 / M) * ONE * CLAT * P[J]))
                    Q = numpy.append(Q,(ONE * (CLAT * Q[J] + SLAT / M * P[J])))
                    SL = numpy.append(SL,(SL[M - 1] * CL[1] + CL[M - 1] * SL[1]))
                    CL = numpy.append(CL,(CL[M - 1] * CL[1] - SL[M - 1] * SL[1]))                    
                else:
                    ONE = numpy.sqrt(N * N - M * M)
                    TWO = numpy.sqrt((N - 1.0) ** 2 - M * M) / ONE
                    THREE = (2.0 * N - 1.0) / ONE
                    I = K - N
                    J = K - 2 * N + 1
                    P = numpy.append(P,((N + 1.0) * (THREE * SLAT / N * P[I] - TWO / (N - 1.0) * P[J])))
                    Q = numpy.append(Q,(THREE * (SLAT * Q[I] - CLAT / N * P[I]) - TWO * Q[J]))
                # end
            # end

            # I have already ingerpolated the GH coefficients to the
            # correct fractional year, so don't need the
            #    dgh + T*igh
            # that you see in magfd.m

            ONE = self.dgh[L] * RR

            #
            #     SYNTHESIS OF X, Y AND Z IN GEOCENTRIC COORDINATES
            #

            if M == 0:      # if M == 0 then there is no Hnm term
                X = X + ONE * Q[K]
                Z = Z - ONE * P[K]
                L = L + 1; # advance only one step in coefficient table
            else:          # otherwise, deal with the Hnm term, too
                TWO = self.dgh[L + 1] * RR;
                THREE = ONE * CL[M] + TWO * SL[M];
                X = X + THREE * Q[K];
                Z = Z - THREE * P[K];
                if CLAT > 0: # checking for div-by-zero; CLAT should be nonnegative
                    Y = Y + (ONE * SL[M] - TWO * CL[M]) * M * P[K] / ((N + 1.0) * CLAT); 
                else:
                    Y = Y + (ONE * SL[M] - TWO * CL[M]) * Q[K] * SLAT;
                # end
                L = L + 2 # advance 2, since both Gnm and Hnm were used
            # end
            M = M + 1
        # end
    
        # CONVERSION TO COORDINATE SYSTEM SPECIFIED BY ITYPE

        LX = X * CD + Z * SD  # geodetic local northward
        LZ = Z * CD - X * SD  # geodetic local up
        LY = Y            # geodetic local east
        
        BR = - Z            # spherical radial (up)
        BTheta = - X            # spherical theta (south)
        BPhi = Y            # spherical phi (east)
            
        BZ = BR * SLAT - BTheta * CLAT;   # absolute cartesian Z
        Brho = BR * CLAT + BTheta * SLAT;   # X-Y plane amplitude
        BX = Brho * CL[1] - BPhi * SL[1]; # absolute cartesian X
        BY = Brho * SL[1] + BPhi * CL[1]; # absolute cartesian Y
        
        BTot = math.sqrt(X * X + Y * Y + Z * Z);

        # these are some debugging loops that I put in while
        # testing for equivalency to magfd.m
        #
        #for k in range(1,len(SL)):
        #    print k, SL[k], CL[k]
        #for k in range(len(self.dgh)):
        #    print k, self.dgh[k]
        #for k in range(1,len(P)):
        #    print k, P[k], Q[k]
        
        return [[LX, LY, LZ], [BR, BTheta, BPhi], [BX, BY, BZ], BTot]

# END

##### testing #####

def igrf_test():
    '''do a quick test of the IGRF model, for the year 2001
    altitude = 100 km, latitude = 60N, longitude = 240E = -120 W
    also, reset to a different year (1902), as well as an invalid
    year (1875) looking for the exception to be raised.'''
    a = IGRF(2010.0)
    c = a.B(100, 60, 240)
    print c

    a.load_gh(1902) # reset coeffiecients to year 1902
    a.load_gh(1875) # should raise exception



if __name__ == '__main__':
    igrf_test()
    
