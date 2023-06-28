import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

t=np.linspace(0,3,12*1024)

l=[220,440,392,196,130.81,146.83,164.81]
a=[0,0.6,1.2,1.8,2.8,3,3.1]
b=[0.1,0.8,1.3,2,3,3.3,3.5]
p=0
j=0
while(j<7):
    p=p+np.multiply(np.sin(2*np.pi*l[j]*t),np.where(np.logical_and((t>a[j]),(t<=b[j])),1,0))
    j=j+1
    
plt.plot(t,p)
sd.play(p,3*1024)


N=3*1024 
f=np.linspace(0,512,int(N/2))
x_f=fft(p)
x_f=2/N* np.abs(x_f[0:np.int(N/2)])
fx=np.random.randint(0,512,2)[0]
fy=np.random.randint(0,512,2)[1]
noise=np.sin(2*fx*np.pi*t)+np.sin(2*fy*np.pi*t)
xnoise=p+noise
xnoise_f=fft(xnoise)
xnoise_f=2/N*np.abs(xnoise_f[0:np.int(N/2)])
maxX_F=max(x_f)
foundfrequenciesindex=np.where(xnoise_f>np.ceil(maxX_F))
maximumfrequencies=f[foundfrequenciesindex]
xfiltered=xnoise
for freq in maximumfrequencies:
    xfiltered=xfiltered - np.sin(2*np.round(freq)*np.pi*t)
    xfilteredf=fft(xfiltered)
    xfilteredf=2/N*np.abs(xfilteredf[0:np.int(N/2)])  
              
plt.subplot(3,2,1)
plt.plot(t,p)

plt.subplot(3,2,2)
plt.plot(f,x_f)

plt.subplot(3,2,3)
plt.plot(t,xnoise)

plt.subplot(3,2,4)
plt.plot(f,xnoise_f)

plt.subplot(3,2,5)
plt.plot(t,xfiltered)

plt.subplot(3,2,6)
plt.plot(f,xfilteredf)

sd.play(xfiltered, 3*1024)
