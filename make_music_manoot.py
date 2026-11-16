import numpy as np, wave, sys
sr=44100; beat=0.5; TOT=30.5
N=int(TOT*sr); L=np.zeros(N); R=np.zeros(N); rng=np.random.default_rng(11)
def add(sig,t,pan=0.0,gain=1.0):
    i=int(t*sr)
    if i>=N or i<0: return
    s=sig[:N-i]; L[i:i+len(s)]+=s*gain*(1-max(pan,0)); R[i:i+len(s)]+=s*gain*(1+min(pan,0))
tt=lambda d: np.arange(int(d*sr))/sr
def ff(x,lo=None,hi=None):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr); m=np.ones_like(f)
    if lo: m*=(f>=lo)
    if hi: m*=(f<=hi)
    return np.fft.irfft(X*m,len(x))
def lp(x,fc):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr); return np.fft.irfft(X/(1+(f/fc)**4),len(x))
def kick():
    t=tt(0.35); f=45+110*np.exp(-t*28); return np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-t*9)
def clap():
    t=tt(0.22); n=ff(rng.standard_normal(len(t)),lo=900,hi=6000); e=np.exp(-t*22)*(1+0.6*(t<0.012)+0.4*((t>0.02)&(t<0.03)))
    return n*e*0.45
def shaker():
    t=tt(0.09); return ff(rng.standard_normal(len(t)),lo=6000)*np.exp(-t*45)*0.16
def bass(f,d):
    t=tt(d); s=np.sin(2*np.pi*f*t)+0.25*np.sin(4*np.pi*f*t); return lp(s,350)*np.minimum(1,t*80)*np.exp(-t*2.5)*0.7
def marimba(f,d=0.5):
    t=tt(d); return (np.sin(2*np.pi*f*t)+0.35*np.sin(2*np.pi*4*f*t)*np.exp(-t*30)+0.2*np.sin(2*np.pi*10*f*t)*np.exp(-t*60))*np.exp(-t*7)*0.22
def pad(freqs,d,g=0.18):
    t=tt(d); s=np.zeros(len(t))
    for f in freqs:
        for det in (-0.6,0,0.6): s+=np.sin(2*np.pi*(f+det)*t)+0.3*np.sin(2*np.pi*2*(f+det)*t)
    s=lp(s,1500)/len(freqs)/3; return s*np.minimum(1,t/0.8)*np.minimum(1,(d-t)/0.8)*g
def whoosh(d=0.5):
    t=tt(d); return ff(rng.standard_normal(len(t)),lo=700,hi=6000)*((t/d)**2)*0.45*np.minimum(1,(d-t)*30)
def riser(d):
    t=tt(d); n=ff(rng.standard_normal(len(t)),lo=500); f=300+2200*(t/d)**2
    return (n*0.5+np.sin(2*np.pi*np.cumsum(f)/sr)*0.35)*(t/d)**2.2*0.5
def boom(d=1.6):
    t=tt(d); f=65*np.exp(-t*1.2)+30; return (np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-t*2.4)+lp(rng.standard_normal(len(t)),300)*np.exp(-t*5)*0.5)*0.8
def chime(d=2.0):
    t=tt(d); return sum(np.sin(2*np.pi*f*t)*np.exp(-t*k) for f,k in ((1568,3),(2093,4),(2637,5)))*0.06
# C - Am - F - G  (bright, tropical)
ch=[([261.63,329.63,392],65.41,[523.25,659.25,783.99,659.25]),([220,261.63,329.63],55.0,[440,523.25,659.25,523.25]),
    ([174.61,220,261.63],43.65,[349.23,440,523.25,440]),([196,246.94,293.66],49.0,[392,493.88,587.33,493.88])]
for b in range(int(TOT/2)+1): add(pad(ch[b%4][0],2.8),b*2.0,0,0.8 if 6.5<=b*2<26.5 else 0.6)
# 0-2 montage: pulse + riser
add(riser(2.0),0,0,0.7)
for k in range(8): add(kick(),k*0.25,0,0.35+0.05*k); add(shaker(),k*0.25+0.125,0.3,0.8)
add(boom(),2.0,0,0.8); add(chime(),2.5,0.2,1.0)
# hero 2.5-6.5: marimba sparse + riser
for k in range(8):
    c=ch[(int((2.5+k*beat)/2))%4]; add(marimba(c[2][k%4]),2.5+k*beat,-0.2,0.8)
add(riser(1.0),5.5,0,0.6)
def groove(t0,t1,dense=False):
    t=t0; step=0
    while t<t1-1e-6:
        c=ch[int(t/2.0)%4]; bi=int(round(((t-t0)%2.0)/beat))
        add(kick(),t,0,0.9)
        if bi in (1,3): add(clap(),t,0,0.8)
        add(bass(c[1]*(2 if bi==2 else 1),beat*0.9),t,0,0.9)
        for q in range(4 if dense else 2):
            dt=beat/(4 if dense else 2); add(shaker(),t+q*dt,0.3,0.7)
        notes=c[2]
        add(marimba(notes[step%4]),t,-0.25,0.9); add(marimba(notes[(step+2)%4]),t+beat*0.75,0.25,0.5)
        step+=1; t+=beat
groove(6.5,22.5)
for tc in np.arange(6.5,22.5,2.0): add(whoosh(0.4),tc-0.4,0,0.5)
add(riser(1.0),21.5,0,0.6)
groove(22.5,26.5,dense=True)
for k in range(8): add(whoosh(0.2),22.5+k*0.5-0.2,0,0.3)
add(riser(0.8),25.7,0,0.6)
add(boom(2.2),26.5,0,0.9); add(chime(3),26.5,0.2,1.2)
for k in range(6):
    c=ch[0]; add(marimba(c[2][k%4]*(1 if k<4 else 2)),26.5+k*beat,-0.2,0.6)
add(pad([261.63,329.63,392,523.25],3.5,0.22),26.5)
m=max(np.abs(L).max(),np.abs(R).max()); L/=m; R/=m
fi=int((TOT-1.5)*sr); fd=np.ones(N); fd[fi:]=np.linspace(1,0,N-fi)
L=np.tanh(L*0.85*fd*1.3)/np.tanh(1.3); R=np.tanh(R*0.85*fd*1.3)/np.tanh(1.3)
w=wave.open(sys.argv[1],"wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.stack([L,R],1)*32767).astype(np.int16).tobytes()); w.close()
