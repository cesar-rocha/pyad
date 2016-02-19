import numpy as np
import spectral_model
from numpy import pi, cos, sin

class RWModel(spectral_model.TwoDimensionalModel):
    """ A subclass that represents the advection-diffusion
            model driven by a renovating waves velocity field

        Attributes
        ----------

    """

    def __init__(self,
                 nmin = 1,
                 nmax = 64,
                 urms=1,
                 power=3,
                 tau = 1,
                 G = 1.,
                 **kwargs):


        # velocity field
        self.nmin = nmin
        self.nmax = nmax
        self.urms = urms
        self.power = power

        self.tau = tau

        super(RWModel, self).__init__(**kwargs)

        # background Q
        self.G = G
        self.Q = self.G*self.y

    def _init_velocity(self):

        self.ntau = int(self.tau/self.dt/2)  # half a cycle
        self.n = np.arange(self.nmin,self.nmax+1)[np.newaxis,np.newaxis,...]
        An = (self.n/self.nmin)**(-self.power/2.)
        N = 2*self.urms/( np.sqrt( ((self.n/self.nmin)**-self.power).sum() ) )
        self.An = N*An

        # estimate the Batchelor scale
        self.S = np.sqrt( ((self.An*self.n*self.dk)**2).sum()/2. )
        self.lb = np.sqrt(self.kappa/self.S)

        #assert self.lb > self.dx, "**Warning: Batchelor scale not resolved."

    def _velocity(self):
        pass
        if (self.ndt%self.ntau == 0.):

            # have this in subclasses
            phase = 2*np.pi*np.random.rand(self.nmax+1-self.nmin)

            #Yn = self.n*self.y[...,np.newaxis] + phase[np.newaxis,...]
            #Xn = self.n*self.x[...,np.newaxis] + phase[np.newaxis,...]

            Xn = self.n*self.x[...,np.newaxis]
            Yn = self.n*self.y[...,np.newaxis]
            self.u = ((self.An*cos(Yn*self.dl + phase[np.newaxis,np.newaxis,...])).sum(axis=2))
            self.v = ((self.An*cos(Xn*self.dk + phase[np.newaxis,np.newaxis,...])).sum(axis=2))

            if self.dirx:
                self.v = self.v*0.
                self.vh = 0.
                self.dirx = False
            else:
                self.u = self.u*0
                self.vh = self.fft2(self.v)
                self.dirx = True

        else:
            pass
