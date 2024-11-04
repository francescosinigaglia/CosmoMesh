import numpy as np

input_filename = '...'
out_filename = '...'

prec = 'single'

ngrid = 576
ngridlow = 360

offset = (ngrid-ngridlow)//2
cent = ngrid//2

if prec == 'single':
    raw = np.fromfile(input_filename, dtype=np.float32)
elif prec	== 'double':
    raw = np.fromfile(input_filename, dtype=np.float64)
else:
    print("Error: this case doesn't exist, check your ode. Exiting.")

raw = np.reshape(raw, (ngrid,ngrid,ngrid))

ft = np.fft.fftn(raw)

#ftnew = ft[offset:ngrid-offset,offset:ngrid-offset,offset:ngrid-offset]
ftnew = np.delete(ft, np.arange(cent-offset,cent+offset), axis=0)
ftnew =	np.delete(ftnew, np.arange(cent-offset,cent+offset), axis=1)
ftnew = np.delete(ftnew, np.arange(cent-offset,cent+offset), axis=2)

#print(ft.shape, ftnew.shape)

rawnew = np.fft.ifftn(ftnew)
rawnew = rawnew.real / ngrid**3 * ngridlow**3

rawnew = rawnew.flatten()

#np.save('wn_abacus_n360_manual_ph000.npy', rawnew)
if prec == 'single':
    rawnew.astype('float32').tofile(out_filename)
elif prec == 'double':
    rawnew.astype('float64').tofile(out_filename)
