from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin
import logging

from netCDF4 import Dataset

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    pass

class TwoDimensionalModel(object):

    """ A class that represents the 2D Advection-Diff model """

    def __init__(
            # grid parameters
            self,
            nx = 256,
            ny=None,
            Lx=2*pi,                    # domain size
            Ly=None,                    # width
            # physical parameters
            nu = 0.,
            G = 1.,
            # timestepping parameters
            dt=.0025,               # numerical timestep
            twrite=100,             # interval for cfl and ke printout (in timesteps)
            tmax=100.,              # total time of integration
            filt=True,              # spectral filter flag
            use_fftw=True,
            ntd = 1,                # number of threads for fftw
            tavestart = 5,
            # filter or dealiasing (False for 2/3 rule)
            use_filter=True,
            # saving parameters
            tsave=100,              # interval to save (in timesteps)
            save2disk=True,
            nmin = 1,
            nmax = 64,
            urms=1,
            power=3,
            tau = 1.,
            diagnostics_list='all',
            logfile=None,
            loglevel=1,
            printcadence = 10,
            diagcadence = 1,
            npad=4):

        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        # initilize parameters

        # domain
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/nx
        self.dy = Ly/ny
        self.x = np.arange(0.,Lx,self.dx)
        self.y = np.arange(0.,Ly,self.dy)
        self.x,self.y = np.meshgrid(self.x,self.y)
        self.dS = self.dx*self.dy
        # constant for spectral normalizations
        self.M = self.nx*self.ny
        self.M2 = self.M**2
        # physical
        self.nu = nu
        # background Q
        self.G = G
        self.Q = self.G*self.y

        # time related variables
        self.ntmax = int(np.ceil(tmax/dt))
        self.dt = dt
        self.twrite = twrite
        self.tmax = tmax
        self.t = 0.
        self.ndt = 0
        self.tsave = tsave
        self.tavestart = tavestart
        self.nsave_max = int(np.ceil(self.ntmax/tsave))
        self.save2disk = save2disk
        self.nsave = 0

        # fourier settings
        self._init_kxky()
        self.kappa2 = self.k**2 + self.l**2
        self.kappa = sqrt(self.kappa2)
        self.fnz = self.kappa2 != 0
        self.kappa2i = np.zeros_like(self.kappa2)   # inversion not defined at kappa=0
        self.kappa2i[self.fnz] = self.kappa2[self.fnz]**-1

        self.diagnostics_list = diagnostics_list
        self.logfile = logfile
        self.loglevel=loglevel
        self.printcadence = printcadence
        self.diagcadence = diagcadence
        self.npad = npad

        # logger
        self._initialize_logger()

        # exponential filter or dealising
        self.use_filter = use_filter
        self._init_filter()

        # fftw
        self.use_fftw = use_fftw
        self.ntd = ntd

        # allocate variables
        self._allocate_variables()

        # DFT
        self._initialize_fft()

        # initialize step forward
        #self._init_rk3w()
        self._init_etdrk4()
        # initialize diagnostics
        self._initialize_diagnostics()

        # initialize tracer field
        self.set_q(np.random.randn(self.ny,self.nx))

        # initialize velocity
        self._init_velocity()

        # initialize fno
        if self.save2disk:
            self._init_fno()

        self.dirx = True
        self._velocity()

    def run(self):
        """ step forward until tmax """

        while(self.t < self.tmax):

            self._stepforward()

            if (self.ndt%self.twrite == 0.):
                self._print_status()
            if self.save2disk:
                self._write2disk()

            self._calc_diagnostics()

            self.t += self.dt
            self.ndt += 1

        if self.save2disk:
            self._close_fno()

    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):
        """ Run the model forward until the next snapshot, then yield."""

        tsnapints = np.ceil(tsnapint/self.dt)
        nt = np.ceil(np.floor((self.tmax-tsnapstart)/self.dt+1)/tsnapints)

        while(self.t < self.tmax):

            self._stepforward()
            if (self.ndt%self.twrite == 0.):
                self._print_status()
            if self.t>=tsnapstart and (self.ndt%tsnapints)==0:
                yield self.t
            if self.save2disk:
                self._write2disk()

            self._calc_diagnostics()

            self.t += self.dt
            self.ndt += 1

        if self.save2disk:
            self._close_fno()

        return

    def _stepforward(self):

        """  Updates velocity field and march one step forward """

        self._velocity()
        self._step_etdrk4()
        #self._step_rk3w()

    def _velocity(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _init_velocity(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _allocate_variables(self):
        """ Allocate variables in memory """

        dtype_real = np.dtype('float64')
        dtype_cplx = np.dtype('complex128')
        shape_real = (self.ny, self.nx)
        shape_cplx = (self.ny, self.nx/2+1)

        # vorticity
        self.q  = np.zeros(shape_real, dtype_real)
        self.qh = np.zeros(shape_cplx, dtype_cplx)
        self.qh0 = np.zeros(shape_cplx, dtype_cplx)
        self.qh1 = np.zeros(shape_cplx, dtype_cplx)

        # velocity
        self.u = np.zeros(shape_real, dtype_real)
        self.v = np.zeros(shape_real, dtype_real)
        # nonlinear-term
        #self.nl1h = np.zeros(shape_cplx, dtype_cplx)
        #self.nl2h = np.zeros(shape_cplx, dtype_cplx)

    def _initialize_fft(self):
        # set up fft functions for use later
        if self.use_fftw:
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.ntd,\
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.ntd,\
                            planner_effort='FFTW_ESTIMATE'))
        else:
            self.fft2 =  (lambda x : np.fft.rfft2(x))
            self.ifft2 = (lambda x : np.fft.irfft2(x))

    def _init_kxky(self):
        """ Calculate wavenumbers """

        self.dl = 2.*pi/self.Ly
        self.dk = 2.*pi/self.Lx
        self.ll = self.dl*np.append( np.arange(0.,self.ny/2),
                np.arange(-self.ny/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nx/2+1)
        self.k,self.l = np.meshgrid(self.kk,self.ll)
        self.kj = 1j*self.k
        self.lj = 1j*self.l

    def set_q(self,q):
        """ Initialize tracer """
        self.q = q
        self.qh = self.fft2(self.q)

    def set_uv(self,u,v):
        """ Initialize velocity field """
        self.v = v
        self.u = u

    def _print_status(self):
        """Output some basic stats."""
        if (self.loglevel) and ((self.ndt % self.printcadence)==0):
            self.var = self.spec_var(self.qh)
            self.cfl = self._calc_cfl()
            self.logger.info('Step: %4i, Time: %3.2e, CFL: %3.2e, Variance: %3.2e'
                    , self.ndt,self.t,self.cfl,self.var)            #assert cfl<1., "CFL condition violated"


    def _write2disk(self):
        """ Save to disk """
        if (self.ndt%self.tsave == 0.):
            self.t_save[self.nsave] = self.t
            self.q_save[:,:,self.nsave] = self.ifft2(self.qh)
            #self.ke_save[self.nsave] = self._calc_ke()
            self.var_save[self.nsave] = self.var
            self.nsave += 1

    def jacobian(self):

        """ Compute the Jacobian in conservative form """

        self.q = self.ifft2(self.qh)
        jach = self.kj*self.fft2(self.u*self.q) +\
                self.lj*self.fft2(self.v*(self.q))\
                + self.G*self.vh

        return jach

    def _init_rk3w(self):

        """ This pre-computes coefficients to a low storage implicit-explicit
            Runge Kutta time stepper.

            See Spalart, Moser, and Rogers. Spectral methods for the navier-stokes
                equations with one infinite and two periodic directions. Journal of
                Computational Physics, 96(2):297 - 324, 1991. """

        self.a1, self.a2, self.a3 = 29./96., -3./40., 1./6.
        self.b1, self.b2, self.b3 = 37./160., 5./24., 1./6.
        self.c1, self.c2, self.c3 = 8./15., 5./12., 3./4.
        self.d1, self.d2 = -17./60., -5./12.

        self.Lin = -self.nu*self.kappa2*self.dt
        self.L1 = ( (1. + self.a1*self.Lin)/(1. - self.b1*self.Lin) )
        self.L2 = ( (1. + self.a2*self.Lin)/(1. - self.b2*self.Lin) )
        self.L3 = ( (1. + self.a2*self.Lin)/(1. - self.b3*self.Lin) )

    def _step_rk3w(self):

        self.nl1h = -self.jacobian()
        self.qh = (self.L1*self.qh + self.c1*self.dt*self.nl1h).copy()
        self.qh = self.filt*self.qh

        self.nl2h = self.nl1h.copy()
        self.nl1h = -self.jacobian()
        self.qh = (self.L2*self.qh + self.c2*self.dt*self.nl1h +\
                self.d1*self.dt*self.nl2h).copy()
        self.qh = self.filt*self.qh

        self.nl2h = self.nl1h.copy()
        self.nl1h = -self.jacobian()
        self.qh = (self.L3*self.qh + self.c3*self.dt*self.nl1h +\
                self.d2*self.dt*self.nl2h).copy()
        self.qh = self.filt*self.qh

    def _init_etdrk4(self):

        """ This performs pre-computations for the Expotential Time Differencing
            Fourth Order Runge Kutta time stepper. The linear part is calculated
            exactly.

            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
                Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005. """

        # the exponent for the linear part
        self.c = -self.nu*self.kappa2

        ch = self.c*self.dt
        self.expch = np.exp(ch)
        self.expch_h = np.exp(ch/2.)
        self.expch2 = np.exp(2.*ch)

        M = 32.  # number of points for line integral in the complex plane
        rho = 1.  # radius for complex integration
        r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M))# roots for integral

        #l1,l2 = self.ch.shape
        #LR = np.repeat(ch,M).reshape(l1,l2,M) + np.repeat(r,l1*l2).reshape(M,l1,l2).T
        LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR

        self.Qh   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=2));
        self.f0  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=2) )
        self.fab =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=2) )
        self.fc  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=2) )

    def _step_etdrk4(self):

        self.qh0 = self.qh.copy()

        Fn0 = -self.jacobian()
        self.qh = (self.expch_h*self.qh0 + Fn0*self.Qh)*self.filt
        self.qh1 = self.qh.copy()

        Fna = -self.jacobian()
        self.qh = (self.expch_h*self.qh0 + Fna*self.Qh)*self.filt

        Fnb = -self.jacobian()
        self.qh = (self.expch_h*self.qh1 + ( 2.*Fnb - Fn0 )*self.Qh)*self.filt

        Fnc =  -self.jacobian()

        self.qh = (self.expch*self.qh0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)*self.filt


    def _init_filter(self):
        """ Set spectral filter """

        if self.use_filter:
            cphi=0.65*pi
            wvx=sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
            self.filt = exp(-23.6*(wvx-cphi)**4.)
            self.filt[wvx<=cphi] = 1.
        else:
            # if not use exponential filter,
            #   then dealias using 2/3 rule
            self.filt = np.ones_like(self.kappa2)
            self.filt[self.nx/3:2*self.nx/3,:] = 0.
            self.filt[:,self.ny/3:] = 0.

    # logger
    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)


        if self.logfile:
            fhandler = logging.FileHandler(filename=self.logfile, mode='w')
        else:
            fhandler = logging.StreamHandler()

        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(self.loglevel*10)

        # this prevents the logger to propagate into the ipython notebook log
        self.logger.propagate = False

        self.logger.info(' Logger initialized')

    # init netcdf output file
    def _init_fno(self):
        self.fno = 'model_output.nc'
        self.FNO = Dataset(self.fno,'w', format='NETCDF4')
        time_dim = self.FNO.createDimension('time_dim', self.nsave_max)
        x_dim = self.FNO.createDimension('x_dim', self.nx)
        y_dim = self.FNO.createDimension('y_dim', self.ny)

        self.q_save = self.FNO.createVariable('q','f4',('y_dim','x_dim','time_dim'))
        self.t_save = self.FNO.createVariable('time','f4',('time_dim'))
        #self.ke_save = self.FNO.createVariable('ke','f4',('time_dim'))
        self.var_save = self.FNO.createVariable('var','f4',('time_dim'))

    # close netcdf output file
    def _close_fno(self):
        self.FNO.close()

    # some diagnostics
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u, self.v])).max()*self.dt/self.dx

    def _calc_ke(self):
        ke = .5*self.spec_var(self.kappa*self.ph)
        return ke.sum()

    def _calc_ens(self):
        ens = .5*self.spec_var(self.qh)
        return ens.sum()

    # diagnostics stuff
    ## diagnostic methods
    def _initialize_nakamura(self):
        self.Lmin2 = self.Lx**2
        # this 2 is arbitrary here..
        thmin,thmax = 0.,2*np.pi
        self.dth = 0.1
        self.dth2 = self.dth**2
        self.TH = np.arange(thmin+self.dth/2,thmax-self.dth/2,self.dth)
        self.Leq2 = np.empty(self.TH.size)
        self.L = np.empty(self.TH.size)

    def _calc_Leq2(self):

        q = self.q + self.Q

        q = np.vstack([(q[self.nx-self.nx/self.npad:]),q,\
                        q[:self.nx/self.npad]])
        gradq2 =  np.vstack([self.gradq2[self.nx-self.nx/self.npad:],\
                              self.gradq2,self.gradq2[:self.nx/self.npad]])
        #gradq2 = self.gradq2

        gradq = np.sqrt(gradq2)

        # parallelize this...
        for i in range(self.TH.size):

            self.fth2 = q<=self.TH[i]+self.dth/2
            self.fth1 = q<=self.TH[i]-self.dth/2

            A2 = self.dS*self.fth2.sum()
            A1 = self.dS*self.fth1.sum()
            self.dA = A2-A1

            self.G2 = (gradq2[self.fth2]*self.dS).sum()-\
                      (gradq2[self.fth1]*self.dS).sum()

            self.Leq2[i] = self.G2*self.dA/self.dth2

            self.L[i] = ((gradq[self.fth2]*self.dS).sum()-\
                        (gradq[self.fth1]*self.dS).sum())/self.dth

    def _calc_diagnostics(self):
        if (self.t>=self.dt) and (self.t>=self.tavestart) and (self.t%self.diagcadence):
            self._increment_diagnostics()

    def _initialize_diagnostics(self):

        # Initialization for diagnotics
        self._initialize_nakamura()

        self.diagnostics = dict()

        self._setup_diagnostics()

        if self.diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif self.diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(self.diagnostics_list)

    def _setup_diagnostics(self):
        """Diagnostics setup"""

        self.add_diagnostic('var',
            description='Tracer variance',
            function= (lambda self: self.spec_var(self.qh))
        )

        self.add_diagnostic('KN',
                    description='Nakamura diffusivity',
                    function= (lambda self: (self.Leq2/self.Lmin2)*self.nu)
                )

        self.add_diagnostic('qbar',
            description='x-averaged tracer',
            function= (lambda self: self.qm)
        )

        self.add_diagnostic('grad2_q_bar',
            description='x-averaged gradient square of th',
            function= (lambda self: self.gradq2m)
        )

        self.add_diagnostic('vq2m',
            description='x-averaged triple advective term v th2',
            function= (lambda self: self.vq2m)
            )

        self.add_diagnostic('q2m',
            description='x-averaged  q2',
            function= (lambda self: self.q2m)
            )

        self.add_diagnostic('vqm',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.q).mean(axis=1))
        )  ### cu

        self.add_diagnostic('fluxy',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.q).mean(axis=1))
        )

        self.add_diagnostic('spec',
            description='spec of anomalies about x-averaged flow',
            function= (lambda self: np.abs(self.fft2(
                        self.q-self.q.mean(axis=1)[...,np.newaxis]))**2/self.M2)
        )

    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

    def add_diagnostic(self, diag_name, description=None, units=None, function=None):
        # create a new diagnostic dict and add it to the object array

        # make sure the function is callable
        assert hasattr(function, '__call__')

        # make sure the name is valid
        assert isinstance(diag_name, str)

        # by default, diagnostic is active
        self.diagnostics[diag_name] = {
           'description': description,
           'units': units,
           'active': True,
           'count': 0,
           'function': function, }

    def describe_diagnostics(self):
        """Print a human-readable summary of the available diagnostics."""
        diag_names = self.diagnostics.keys()
        diag_names.sort()
        print('NAME               | DESCRIPTION')
        print(80*'-')
        for k in diag_names:
            d = self.diagnostics[k]
            print('{:<10} | {:<54}').format(
                 *(k,  d['description']))

    def _increment_diagnostics(self):

        self._calc_derived_fields()

        for dname in self.diagnostics:
            if self.diagnostics[dname]['active']:
                res = self.diagnostics[dname]['function'](self)
                if self.diagnostics[dname]['count']==0:
                    self.diagnostics[dname]['value'] = res
                else:
                    self.diagnostics[dname]['value'] += res
                self.diagnostics[dname]['count'] += 1

    def _calc_derived_fields(self):

        """ Calculate derived field necessary for diagnostics """

        # x-averaged tracer field
        self.qm = self.q.mean(axis=1)

        # anomaly about the x-averaged field
        self.qa = self.q -self.qm[...,np.newaxis]*0.
        self.qah = self.fft2(self.qa)

        # x-averaged gradient squared
        gradx = self.ifft2(1j*self.k*self.qah)
        grady = self.ifft2(1j*self.l*self.qah)

        self.gradq2 = (gradx**2 + grady**2)
        self.gradq2m = self.gradq2.mean(axis=1)

        # triple term
        self.vq2m = (self.v*(self.qa**2)).mean(axis=1)

        # diff transport
        self.q2m = (self.qa**2).mean(axis=1)

        # Leq2
        self._calc_Leq2()

    # def _calc_Leq2(self):
    #     raise NotImplementedError(
    #         'needs to be implemented by Model subclass')

    def get_diagnostic(self, dname):
        diag = (self.diagnostics[dname]['value'] /
                 self.diagnostics[dname]['count'])
        return  diag

    def spec_var(self,ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / (self.nx*self.ny)**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
        return var_dens.sum()
