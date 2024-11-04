import numpy as np
from numba import njit, prange

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# General parameters
filename1 = 'field1.dat'     # field1 filename 
filename2 = 'field2.dat'     # field2 filename 

lbox = 500                  # Box size in Mpc/h
ngrid = 128                 # mesh size

convert1_to_delta = True   # If ouput of webon, set to False, as webon produces already the overdensty field
convert2_to_delta = True   # If ouput of webon, set to False, as webon produces already the overdensty field

write_pk = True
outpk1 = 'pk1.txt'
outpk2 = 'pk2.txt'

write_ck = True
outck = 'ck.txt'

prec1 = np.float32
prec2 = np.float32

nbins_pk = ngrid//2

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
# **********************************************
# **********************************************

lcell = lbox/ngrid

# Read input arrays
delta1 = np.fromfile(filename1, dtype=prec1)           # In real space
delta2 = np.fromfile(filename2, dtype=prec2)           # In real space

# Convert DM field to overdensity, if requested. Tracer field must be done later
if convert1_to_delta == True:
    delta1 = delta1/np.mean(delta1) - 1.

if convert2_to_delta == True:
    delta2 = delta2/np.mean(delta2) - 1.

delta1 = np.reshape(delta1, (ngrid,ngrid,ngrid))
delta2 = np.reshape(delta2, (ngrid,ngrid,ngrid))

# Compute P(k) of the reference tracer field

kk1, pk1 = measure_spectrum(delta1)
kk2, pk2 = measure_spectrum(delta2)

kk, ck = cross_spectrum(delta1, delta2)

ck = compute_cross_correlation_coefficient(ck, pk1,pk2)

if write_pk==True:
    ff = open(outpk1, 'w')
    ff.write('# k       P(k) \n')
    
    for ii in range(len(kk1)):
        ff.write(str(kk1[ii]) + '     ' + str(pk1[ii]) + '\n') 

    ff.close()   

    gg = open(outpk2, 'w')
    gg.write('# k       P(k) \n')
    
    for jj in range(len(kk2)):
        gg.write(str(kk2[jj]) + '     ' + str(pk2[jj]) + '\n') 

    gg.close()  

if write_ck == True:

    hh = open(outck, 'w')
    hh.write('# k       C(k) \n')
    
    for ll in range(len(kk)):
        hh.write(str(kk[ll]) + '     ' + str(ck[ll]) + '\n') 

    hh.close()  


