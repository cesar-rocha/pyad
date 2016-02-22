import numpy as np

def calc_Koc(model):
    # calculate the Koc
    thm = model.get_diagnostic('qbar')
    thmh = np.fft.rfft(thm)
    thay = np.fft.irfft(1j*model.kk*thmh)
    cby2 = thay**2

    grad2 =  model.get_diagnostic('grad2_q_bar')

    cby2 = 1.

    Koc = (grad2/cby2)*model.nu
    D = (model.urms**2)*model.tau/4.

    return Koc, D


def variance_budget(model):

    # calculate the Koc
    thm = model.get_diagnostic('qbar')
    thmh = np.fft.rfft(thm)
    thmy = np.fft.irfft(1j*model.kk*thmh)
    cby2 = thmy**2

    cby2 = model.G**2

    th2 = model.get_diagnostic('q2m')
    th2h = np.fft.rfft(th2)
    th2yy = np.fft.irfft(-((model.kk)**2)*th2h)

    grad2 =  model.get_diagnostic('grad2_q_bar')

    vth2 = model.get_diagnostic('vq2m')
    vth2h = np.fft.rfft(vth2)
    vth2y = np.fft.irfft(1j*model.kk*vth2h)

    D = (model.urms**2)*model.dt/4.
    Koc = (grad2/cby2)*model.nu

    prod = -model.get_diagnostic('vqm')*model.G

    var_trans = -vth2y/2.
    #eddy_diff2 = vthm*thmy
    eddy_diff= Koc*(cby2**2)

    diff_trans = model.nu*th2yy/2.
    diff = -model.nu*grad2

    return var_trans, eddy_diff, diff_trans, diff,prod
