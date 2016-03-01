import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
#import spectral_model as model
import pyad as pyad
# import renovating_waves as model
from pyad.utils import *

reload(pyad)

# the model object
m = pyad.RWModel(Lx=2.*np.pi,nx=1024, tmax = 100,tavestart=50.,dt = .00125,
                            use_fftw=True, use_filter=True,nu=1.e-4,
                            save2disk=False,
                            tsave = 400,
                            tau = 1.,
                            G=1.,
                            power=4,
                            nmin=2,
                            nmax = 20,
                            diagcadence = 50,
                            npad = 2,
                            ntd=4,
                            etdrk4=False)

qi = m.y*0
m.set_q(qi)

# m.run()

t, var = [],[]

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=m.dt):
    t.append(m.t)
    var.append(m.spec_var(m.qh))


kappa_num = 1.

Koc, D = calc_Koc(m)

Koc = kappa_num*Koc

var_trans, eddy_diff,  diff_trans, diff, prod = variance_budget(m)

res = var_trans + diff_trans*kappa_num + diff*kappa_num + prod

KN = m.get_diagnostic('KN')*kappa_num

spec = m.get_diagnostic('spec')

np.savez('modelrun_tau_1',Koc=Koc,D=D,y=m.y[...,0],KN=KN,Q=m.TH,spec=spec,k=m.kk,
          l=m.ll, var_trans = var_trans, diff_trans=diff_trans, diss=diff,
          prod=prod, t=np.array(t),var=np.array(var), res=res)

# plt.ion()
#
# t, var = [],[]
#
# for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=10*m.dt):
#
#    #un,vn = uvnoise()
#    #_velocity(m)
#
#
#     t.append(m.t)
#     var.append(m.spec_var(m.qh))
#
#     plt.clf()
#     p1 = plt.imshow(m.q+m.Q)#,np.linspace(-1,1,10))
#     #p1 = plt.contourf(m.q,np.linspace(0,1,10),extend='both')
#     #plt.clim([0, 7.])
#     #plt.contour(m.q,np.linspace(0,4,10),colors='k')
#     #plt.quiver(m.x[::5,::5],m.y[::5,::5],m.u[::5,::5],m.v[::5,::5])
#     #plt.contourf(m.v)
#     plt.title('t='+str(m.t))
#
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.pause(0.1)
#
#     plt.draw()
#
# plt.show()
# plt.ion()
