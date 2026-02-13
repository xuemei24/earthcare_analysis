import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv('aods_aeronet.txt',delimiter='\t')
time = f['Month']   
aeronet = f['AERONET']
aeronet_std = f['AERONET STD']
atlid = f['ATLID']
atlid_std = f['ATLID STD']       

fig,ax = plt.subplots(1,figsize=(6,4))
ax.plot(time,aeronet,color='black',label='AERONET')
ax.fill_between(time,aeronet-aeronet_std,aeronet+aeronet_std,color='black',alpha=0.3)
ax.plot(time,atlid,color='orange',label='ATLID')
ax.fill_between(time,atlid-atlid_std,atlid+atlid_std,color='orange',alpha=0.3)
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('AOD',fontsize=15)
ax.tick_params(labelsize=15)
ax.tick_params(labelsize=15)
ax.legend(frameon=False)
fig.savefig('atlid_aeronet_timeseries.jpg')
