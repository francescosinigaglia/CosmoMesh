import numpy as np
from numba import njit, prange
import scipy.optimize

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# General parameters
tracer_filename = 'tracers.DAT'     # tracer field filename 

dm_filename =  'dm.DAT'        # dark matter field filename

tweb_filename = 'tweb.DAT'       # T-web filename
tdeltaweb_filename = 'tdeltaweb.DAT'  # T-delta-web filename

lbox = 500                  # Box size in Mpc/h
ngrid = 128                 # mesh size

redshift = 2.               # Redshift, not relevant if you donpt have velocities

convert_dm_to_delta = True   # If ouput of webon, set to False, as webon produces already the overdensty field
convert_tr_to_delta = True

# Cosmological parameters
h = 0.6774
H0 = 100
Om = 0.3089
Orad = 0.
Ok = 0.
N_eff = 3.046
w_eos = -1
Ol = 1-Om-Ok-Orad
aa = 1./(1.+redshift)

# Model parameters                                                                                                                                                                                                                        

cwtype = -1
cwdeltatype = -1

bounds = ((0., 1.),(0.,2.))
x0 = (0.2, 1.3)

#bounds_full = ((),(),(),())
#x0_full = 

nbins_pdf = 31
nbins_pk = ngrid//2     # Use by default the naturla gridding

kmin = 0.             # Minimum k used for the fit of the P(k). Set it arbitrarily small (e.g. 0, or even negative) if you don't want a lower limit
kmax = 1e6             # Maximum k used for the fit of the P(k). Set it arbitrarily large (e.g. 1e6) if you don't want an upper limit


pdf_to_pk_chiqsq_ratio = 1. # Relative weight that you want to assign to the chisquare from the PDF. 
                            # E.g., if N,: chisq_tot = chisq_pk + N*chisq_pdf 

# Random seed for stochasticity reproducibility
np.random.seed(123456)

# **********************************************
# **********************************************
# **********************************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************
def measure_spectrum(signal):

    nbin = int(round(nbins_pk))
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]

# **********************************************                                                                                    
def cross_spectrum(signal1, signal2):

    nbin = round(ngrid/2)

    fsignal1 = np.fft.fftn(signal1) #np.fft.fftn(signal)                                             
    fsignal2 = np.fft.fftn(signal2) #np.fft.fftn(signal)                                             

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))            
    dk = kmax/nbin  # Bin width                                                                                                    

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_cross_power(fsignal1, fsignal2, nbin, kmax, dk, kmode, power, nmode)

    return kmode[1:], power[1:]

# **********************************************                                                                                        
def compute_cross_correlation_coefficient(cross, power1,power2):
    ck = cross/(np.sqrt(power1*power2))
    return ck

# **********************************************                                                                                         
@njit(parallel=False, cache=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode):
    
    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode

# **********************************************                                                                            
@njit(parallel=False, cache=True)
def get_cross_power(fsignal1, fsignal2, Nbin, kmax, dk, kmode, power, nmode):

    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl1 = fsignal1.real[i,j,k]
                    bkl1 = fsignal1.imag[i,j,k]
                    akl2 = fsignal2.real[i,j,k]
                    bkl2 = fsignal2.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl1*akl2+bkl1*bkl2)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode
  
# **********************************************
@njit(parallel=True, cache=True)
def get_cic(posx, posy, posz, lbox, ngrid):

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(ngrid**3):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lbox)
        indyc = int(yy/lbox)
        indzc = int(zz/lbox)

        wxc = xx/lbox - indxc
        wyc = yy/lbox - indyc
        wzc = zz/lbox - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            wxl = 1 - wxc

        if wyc <=0.5:
            indyl = indyc - 1
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            wyl = 1 - wyc

        if wzc <=0.5:
            indzl = indzc - 1
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            wzl = 1 - wzc

        delta[indxc,indyc,indzc] += wxc*wyc*wzc
        delta[indxl,indyc,indzc] += wxl*wyc*wzc
        delta[indxc,indyl,indzc] += wxc*wyl*wzc
        delta[indxc,indyc,indzl] += wxc*wyc*wzl
        delta[indxl,indyl,indzc] += wxl*wyl*wzc
        delta[indxc,indyl,indzl] += wxc*wyl*wzl
        delta[indxl,indyc,indzl] += wxl*wyc*wzl
        delta[indxl,indyl,indzl] += wxl*wyl*wzl

    delta = delta/np.mean(delta) - 1.

    return delta

# **********************************************
# Real to redshift space mapping
#@njit(parallel=False, cache=True)
def real_to_redshift_space(delta, vel, tweb, ngrid, lbox) :

    H = H0*h*np.sqrt(Om*(1+zz)**3 + Ol)

    xx = np.repeat(np.arange(ngrid), ngrid**2*num_part_per_cell)*lcell
    yy = np.tile(np.repeat(np.arange(ngrid), ngrid *num_part_per_cell), ngrid)*lcell


    #RSD model
    velarr = np.repeat(vel,num_part_per_cell)

    # Coherent flows
    velbias = 0.*delta.copy()
    velbias[np.where(tweb==1)] = B1
    velbias[np.where(tweb==2)] = B2
    velbias[np.where(tweb==3)] = B3
    velbias[np.where(tweb==4)] = B4
    velbias = np.repeat(velbias, num_part_per_cell) 

    # Quasi-virialized motions
    sigma = 0.*delta.copy()
    sigma[np.where(tweb==1)] = b1*(1. + delta[np.where(tweb==1)])**beta1
    sigma[np.where(tweb==2)] = b2*(1. + delta[np.where(tweb==2)])**beta2
    sigma[np.where(tweb==3)] = b3*(1. + delta[np.where(tweb==3)])**beta3
    sigma[np.where(tweb==4)] = b4*(1. + delta[np.where(tweb==4)])**beta4
    sigma = np.repeat(sigma, num_part_per_cell)

    rand = np.random.normal(0,sigma)
    velarr = velarr + rand

    zz = np.tile(np.repeat(np.arange(ngrid), num_part_per_cell), ngrid**2)*lcell + velbias*velarr/(aa*H)

    # Periodic boundaries:
    zz[np.where(zz>lbox)] = zz[np.where(zz>lbox)] - lbox
    zz[np.where(zz<0)] = zz[np.where(zz<0)] + lbox

    delta = get_cic(xx, yy, zz, lbox, ngrid)

    return delta


# **********************************************
@njit(parallel=True, cache=True)
def divide_by_k2(delta,ngrid, lbox):
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in prange(round(ngrid/2.)): 
                k2 = k_squared(lbox,ngrid,ii,jj,kk) 
                if k2>0.:
                    delta[ii,jj,kk] /= -k_squared(lbox,ngrid,ii,jj,kk) 

    return delta
    

# **********************************************
#@njit(parallel=True, cache=True)
def poisson_solver(delta, ngrid, lbox):

    delta = fftr2c(delta)

    delta = divide_by_k2(delta, ngrid, lbox)
    
    delta[0,0,0] = 0.

    delta = fftc2r(delta)

    return delta

# **********************************************
@njit(cache=True)
def k_squared(lbox,ngrid,ii,jj,kk):
    
      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)
      
      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)
      
      #if kk <= nc/2:
      kz = kfac*kk
      #else:
      #  kz = -kfac*np.float64(nc-k)
      
      k2 = kx**2+ky**2+kz**2

      return k2

@njit(cache=True)
def k_squared_nohermite(lbox,ngrid,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)

      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)

      if kk <= ngrid/2:
          kz = kfac*kk
      else:
          kz = -kfac*(ngrid-kk)                                                                                                           

      k2 = kx**2+ky**2+kz**2

      return k2

# **********************************************
@njit(parallel=True, cache=True)
def gradfindiff(lbox,ngrid,arr,dim):

    fac = ngrid/(2*lbox)

    outarr = arr.copy()

    for xx in prange(ngrid):
        for yy in prange(ngrid):
            for zz in prange(ngrid):

                xdummy = np.array([xx,xx,xx,xx])
                ydummy = np.array([yy,yy,yy,yy])
                zdummy = np.array([zz,zz,zz,zz])
                xxr = xdummy[0]
                xxrr = xdummy[1]
                xxl = xdummy[2]
                xxll = xdummy[3]
                yyr = ydummy[0]
                yyrr = ydummy[1]
                yyl = ydummy[2]
                yyll = ydummy[3]
                zzr = zdummy[0]
                zzrr = zdummy[1]
                zzl = zdummy[2]
                zzll = zdummy[3]

                # Periodic BCs
                if dim == 1:
                    xxl = xx - 1
                    xxll = xx - 2
                    xxr = xx + 1
                    xxrr = xx + 2
                    
                    if xxl<0:
                        xxl += ngrid
                    if xxl>=ngrid:
                        xxl -= ngrid
                    
                    if xxll<0:
                        xxll += ngrid
                    if xxll>=ngrid:
                        xxll -= ngrid
                    
                    if xxr<0:
                        xxr += ngrid
                    if xxr>=ngrid:
                        xxr -= ngrid

                    if xxrr<0:
                        xxrr += ngrid
                    if xxrr>=ngrid:
                        xxrr -= ngrid


                elif dim == 2:
                    
                    yyl = yy - 1
                    yyll = yy - 2
                    yyr = yy + 1
                    yyrr = yy + 2
                    
                    if yyl<0:
                        yyl += ngrid
                    if yyl>=ngrid:
                        yyl -= ngrid
                    
                    if yyll<0:
                        yyll += ngrid
                    if yyll>=ngrid:
                        yyll -= ngrid
                    
                    if yyr<0:
                        yyr += ngrid
                    if yyr>=ngrid:
                        yyr -= ngrid

                    if yyrr<0:
                        yyrr += ngrid
                    if yyrr>=ngrid:
                        yyrr -= ngrid


                elif dim == 3:
                    
                    zzl = zz - 1
                    zzll = zz - 2
                    zzr = zz + 1
                    zzrr = zz + 2
                    
                    if zzl<0:
                        zzl += ngrid
                    if zzl>=ngrid:
                        zzl -= ngrid
                    
                    if zzll<0:
                        zzll += ngrid
                    if zzll>=ngrid:
                        zzll -= ngrid
                    
                    if zzr<0:
                        zzr += ngrid
                    if zzr>=ngrid:
                        zzr -= ngrid

                    if zzrr<0:
                        zzrr += ngrid
                    if zzrr>=ngrid:
                        zzrr -= ngrid

                outarr[xx,yy,zz] = -fac*((4.0/3.0)*(arr[xxl,yyl,zzl]-arr[xxr,yyr,zzr])-(1.0/6.0)*(arr[xxll,yyll,zzll]-arr[xxrr,yyrr,zzrr]))

    return outarr

# **********************************************
@njit(parallel=True, cache=True)
def get_tidal_invariants(arr, ngrid, lbox):

    # Get gradients exploiting simmetry of the tensor, i.e. gradxy=gradyx

    # X DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,1)
    #2nd derivs
    gradxx = gradfindiff(lbox,ngrid,grad,1)
    gradxy = gradfindiff(lbox,ngrid,grad,2)
    gradxz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,2)
    #2nd derivs
    gradyy = gradfindiff(lbox,ngrid,grad,2)
    gradyz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,3)
    #2nd derivs
    gradzz = gradfindiff(lbox,ngrid,grad,3)

    #del arr, grad

    lambda1 = np.zeros_like((gradxx))
    lambda2 = np.zeros_like((gradxx))
    lambda3 = np.zeros_like((gradxx))
    tweb = np.zeros_like((gradxx))

    # Compute eigenvalues    
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in prange(ngrid):
                mat = np.array([[gradxx[ii,jj,kk],gradxy[ii,jj,kk],gradxz[ii,jj,kk]],[gradxy[ii,jj,kk],gradyy[ii,jj,kk],gradyz[ii,jj,kk]],[gradxz[ii,jj,kk],gradyz[ii,jj,kk],gradzz[ii,jj,kk]]])
                eigs = np.linalg.eigvals(mat)
                eigs = np.flip(np.sort(eigs))
                lambda1[ii,jj,kk] = eigs[0]
                lambda2[ii,jj,kk] = eigs[1]
                lambda3[ii,jj,kk] = eigs[2]
                if eigs[0]>=0 and eigs[1]>=0 and eigs[2]>=0:
                    tweb[ii,jj,kk] = 1
                elif eigs[0]>=0 and eigs[1]>=0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 2
                elif eigs[0]>=0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 3
                elif eigs[0]<0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 4

    # Now compute invariants
    #del gradxx, gradxy, gradxz, gradyy,gradyz,gradzz
    
    #I1 = lambda1 + lambda2 + lambda3
    #I2 = lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3
    #I3 = lambda2 * lambda2 * lambda3

    #del lambda1, lambda2, lambda3

    return tweb

# **********************************************
#@njit(parallel=True, cache=True)

def chisquare(xx):
    
    tr_new = tr_ref.copy()
    deltatemp = delta.copy()
    
    if cwtype > 0 and cwdeltatype > 0:
        indd = np.where(np.logical_and(tweb==cwtype,tdeltaweb==cwdeltatype))
    
    elif cwtype > 0 and cwdeltatype < 0:
       indd = np.where(tweb==cwtype)

    elif cwtype < 0:
        indd = np.where(tweb>0)

    else:
        print('Error: no such a case should exists. Exiting.')
        exit()

    # Write here the model 
    # Model for partial fit
    tr_new[indd] = xx[0] * (1+deltatemp[indd])**xx[1]
    
    """
    # Model for full fit
    if cwtype > 0 and cwdeltatype > 0:
        indfull = np.where(np.logical_and(tweb==1, tdeltaweb==1))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==1, tdeltaweb==2))
        tr_new[indfull] = xx[4] * (1+deltatemp[indfull])**xx[5] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==1, tdeltaweb==3))
        tr_new[indfull] = xx[6] * (1+deltatemp[indfull])**xx[9] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==1, tdeltaweb==4))
        tr_new[indfull] = xx[12] * (1+deltatemp[indfull])**xx[13] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])

        indfull = np.where(np.logical_and(tweb==2, tdeltaweb==1))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==2, tdeltaweb==2))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==2, tdeltaweb==3))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==2, tdeltaweb==4))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])

        indfull = np.where(np.logical_and(tweb==3, tdeltaweb==1))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==3, tdeltaweb==2))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==3, tdeltaweb==3))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==3, tdeltaweb==4))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])

        indfull = np.where(np.logical_and(tweb==4, tdeltaweb==1))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==4, tdeltaweb==2))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==4, tdeltaweb==3))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])
        indfull = np.where(np.logical_and(tweb==4, tdeltaweb==4))
        tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1] * np.exp(-deltatemp[indfull]/xx[2]) * np.exp(deltatemp[indfull]/xx[3])

    #elif cwtype > 0 and cwdeltatype < 0:
    indfull = np.where(tweb==1)
    tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1]
    indfull = np.where(tweb==2)
    tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1]
    indfull = np.where(tweb==3)
    tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1]
    indfull = np.where(tweb==4)
    tr_new[indfull] = xx[0] * (1+deltatemp[indfull])**xx[1]

    else:
        tr_new = xx[0] * (1+deltatemp)**xx[1] * np.exp(-deltatemp/xx[2]) * np.exp(deltatemp/xx[3])
    """
        
    # Now make diagnostics of the new tracer field
    bins = np.linspace(np.amin(tr_ref),np.amax(tr_ref),num=nbins_pdf)
    hh, edg = np.histogram(tr_new.flatten(), bins=bins, density=True)

    # Pass to tracer overdensity  
    tr_new = tr_new/np.mean(tr_new) - 1.

    # Measure the P(k) of the new tracer field
    kknew, pknew =  measure_spectrum(tr_new)

    ind_pk = np.where(np.logical_and(kknew<kmax,kknew>kmin))
    chisqpk = np.sum(( pknew[ind_pk] / pkref[ind_pk] - 1.)**2) / len(pknew[ind_pk])
    chisqpdf = np.sum((hh/hhref - 1)**2)/len(hh)

    chisq = chisqpk + pdf_to_pk_chiqsq_ratio * chisqpdf

    print('-------------------------------')
    print('Total chi square: ', chisq)
    print('P(k) chisquare: ', chisqpk)
    print('PDF chisquare: ', chisqpdf)
    print('P(k)/PDF chisquare ratio', chisqpk/chisqpdf)
    
    print('P(k) residuals:')
    print(pknew[ind_pk]/pkref[ind_pk] - 1.)

    print('')
    print('Parameters at this iteration: ', xx)
    
    print('-------------------------------')
    print('')

    return chisq
    

def optimizer():

    #bounds = bounds#((0.95, 1.05),(0.,2.),(-0.2,0.),(0.1, 3.), (0.,5), (0.,5))

    #kkref, pkref = measure_spectrum(fluxref)
    
    #res = scipy.optimize.differential_evolution(chisquare,bounds, tol=1e-4, x0=x0)
    minimizer_kwargs = {"bounds": bounds}
    res = scipy.optimize.basinhopping(chisquare, x0, niter=2000, minimizer_kwargs=minimizer_kwargs)
    
    return res['x']
    

# **********************************************
# **********************************************
# **********************************************

lcell = lbox/ngrid

aa = 1./(1.+redshift)

# Read input arrays
tr_ref = np.fromfile(open(tracer_filename, 'r'), dtype=np.float32)      # In real space
delta = np.fromfile(open(dm_filename, 'r'), dtype=np.float32)           # In real space
tweb = np.fromfile(open(tweb_filename, 'r'), dtype=np.float32)          # In real space
tdeltaweb = np.fromfile(open(tdeltaweb_filename, 'r'), dtype=np.float32)     # In real space

# Convert DM field to overdensity, if requested. Tracer field must be done later
if convert_dm_to_delta == True:
    delta = delta/np.mean(delta) - 1.

delta = np.reshape(delta, (ngrid,ngrid,ngrid))
tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))
tdeltaweb = np.reshape(tdeltaweb, (ngrid,ngrid,ngrid))

# Compute reference tracer field PDF
binsref = np.linspace(np.amin(tr_ref),np.amax(tr_ref),num=nbins_pdf)
hhref, edgref = np.histogram(tr_ref, bins=binsref, density=True)

# Convert tracer field to overdensity, if requested
if convert_tr_to_delta == True:
     tr_ref = tr_ref/np.mean(tr_ref) - 1.

tr_ref = np.reshape(tr_ref, (ngrid,ngrid,ngrid))

# Compute P(k) of the reference tracer field

kkref, pkref = measure_spectrum(tr_ref)

pars = optimizer()
