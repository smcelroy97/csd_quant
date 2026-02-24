'''
Eventual use is to load a directory of LFP or CSD files to
then be downsampled, flattened, and run PCA to create an
"ideal" case CSD patten for an auditory ERP as seen in
Rimehaug et al. (2023) https://doi.org/10.7554/eLife.87169
'''

import h5py
import scipy
import numpy as np

fn = 'Arnaud_aSSR data/1-WI03022@oe.mat'
samprds = 200

fp = h5py.File(fn, 'r')

sampr = fp['adrate'][0][0]  # sampling rate
dat = fp['cnt']  # cnt record stores the electrophys data
print('fn:', fn, 'sampr:', sampr, 'samprds:', samprds)

dt = 1.0 / sampr  # time-step in seconds
npdat = np.zeros(dat.shape)
tmax = (len(npdat) - 1.0) * dt  # use original sampling rate for tmax - otherwise shifts phase
dat.read_direct(npdat)  # read it into memory; note that this LFP data usually stored in microVolt
npdat *= 0.001  # convert microVolt to milliVolt here
fp.close()

siglen = max((npdat.shape[0], npdat.shape[1]))
nchan = min((npdat.shape[0], npdat.shape[1]))
npds = []  # zeros((int(siglen/float(dsfctr)),nchan))
# print('npdat.shape:',npdat.shape)
for i in range(nchan):
    print('resampling channel', i)
    npds.append(scipy.signal.resample(npdat[:, i], int(siglen * samprds / sampr)))
# print(dsfctr, dt, siglen, nchan, samprds, ceil(int(siglen / float(dsfctr))), len(npds),len(npds[0]))
npdat = np.array(npds)
npdat = npdat.T
sampr = samprds


