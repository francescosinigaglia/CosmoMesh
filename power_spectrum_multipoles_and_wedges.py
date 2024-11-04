import numpy as np
from numba import njit, prange

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# General parameters
filename = 'field.dat'     # field filename 

lbox = 500                  # Box size in Mpc/h
ngrid = 128                 # mesh size

convert_to_delta = True   # If ouput of webon, set to False, as webon produces already the overdensty field

write_pk = True
outpk = 'pk.txt'

prec = np.float32

nbins_pk = ngrid//2

mumin = 0.
mumax = 1.

axis = 2

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
    mono = np.zeros((nbin))
    quadru = np.zeros((nbin))
    hexa = np.zeros((nbin))

    kmode, monopole, quadrupole, hexadecapole, nmode = get_power(fsignal, nbin, kmax, dk, kmode, mono, quadru, hexa, nmode, mumin, mumax, axis)

    return kmode[1:], monopole[1:], quadrupole[1:], hexadecapole[:1]

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
def get_power(fsignal, Nbin, kmax, dk, kmode, mono, quadru, hexa, nmode, mumin, mumax, axis):
    
    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))

                if ktot <= kmax:

                    kpar, kper = get_k_par_per(lbox,ngrid,ii,jj,kk)

                    # find the value of mu
                    if ii==0 and jj==0 and kk==0:  
                        mu = 0.0
                    else:    
                        mu = k_par/ktot
                    mu2 = mu*mu

                    # take the absolute value of k_par
                    if k_par<0:  k_par = -k_par
                    
                    if mu>=mumin and mu<mumax:
                        nbin = int(ktot/dk-0.5)
                        akl = fsignal.real[i,j,k]
                        bkl = fsignal.imag[i,j,k]
                        kmode[nbin]+=ktot
                        delta2 = akl*akl+bkl*bkl
                        mono[nbin] += delta2
                        quadru[ii] += delta2*(3.0*mu2-1.0)/2.0
                        hexa[ii]   += delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0
                        nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            mono[m]/=nmode[m]
            quadru[m]/=nmode[m]*5.  # we need to multiply the multipoles by (2*ell + 1)
            hexa[m]/=nmode[m]*9.    # we need to multiply the multipoles by (2*ell + 1)

    power = power / (ngrid/2)**3

    return kmode, mono, quadru, hexa, nmode

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

@njit(cache=True)
def get_k_par_per(lbox,ngrid,ii,jj,kk):

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

    # compute the value of k_par and k_perp
    if axis==0:   
        k_par, k_per = kx, np.sqrt(ky*ky + kz*kz)
    elif axis==1: 
        k_par, k_per = ky, np.sqrt(kx*kx + kz*kz)
    else:         
        k_par, k_per = kz, np.sqrt(kx*kx + ky*ky)
                                                                                                               
    return k_par, k_per

# **********************************************
# **********************************************
# **********************************************

lcell = lbox/ngrid

# Read input arrays
delta = np.fromfile(filename, dtype=prec)           # In real space

# Convert DM field to overdensity, if requested. Tracer field must be done later
if convert_to_delta == True:
    delta = delta/np.mean(delta) - 1.

delta = np.reshape(delta, (ngrid,ngrid,ngrid))

# Compute P(k) of the reference tracer field

kk, pk_l0, pk_l2, pk_l4 = measure_spectrum(delta)

if write_pk==True:
    ff = open(outpk, 'w')
    ff.write('# k       P(k) (l=0)    P(k) (l=4)    P(k) (l=4) \n')
    
    for ii in range(len(kk)):
        ff.write(str(kk[ii]) + '     ' + str(pk_l0[ii]) + '     ' + str(pk_l2[ii]) + '     ' + str(pk_l4[ii]) +'\n') 

    ff.close()   
