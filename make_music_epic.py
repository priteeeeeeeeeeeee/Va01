import numpy as np, wave, sys
sr=44100; beat=0.5; TOT=27.0
N=int(TOT*sr); L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(5)
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
    t=tt(0.4); f=42+120*np.exp(-t*26); return np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-t*8)
def snare():
    t=tt(0.25); n=ff(rng.standard_normal(len(t)),lo=1500); return n*np.exp(-t*18)*0.5+np.sin(2*np.pi*190*t)*np.exp(-t*25)*0.4
def hat(o=False):
    t=tt(0.3 if o else 0.12); return ff(rng.standard_normal(len(t)),lo=7000)*np.exp(-t*(14 if o else 60))*0.25
def bass(f,d):
    t=tt(d); s=np.sin(2*np.pi*f*t)+0.35*np.sin(4*np.pi*f*t); return lp(s,420)*np.minimum(1,t*80)*np.minimum(1,(d-t)*40)*0.65
def pluck(f,d=0.4):
    t=tt(d); return sum(np.sin(2*np.pi*f*k*t)/k for k in (1,2,3))*np.exp(-t*9)*0.17
def pad(freqs,d,g=0.22):
    t=tt(d); s=np.zeros(len(t))
    for f in freqs:
        for det in (-0.5,0,0.5): s+=np.sign(np.sin(2*np.pi*(f+det)*t))*0.5+np.sin(2*np.pi*(f+det)*t)
    s=lp(s,1100)/len(freqs)/3; return s*np.minimum(1,t/0.7)*np.minimum(1,(d-t)/0.7)*g
def whoosh(d=0.5):
    t=tt(d); return ff(rng.standard_normal(len(t)),lo=800,hi=6500)*((t/d)**2)*0.5*np.minimum(1,(d-t)*30)
def riser(d):
    t=tt(d); n=ff(rng.standard_normal(len(t)),lo=500); f=200+2600*(t/d)**2
    return (n*0.5+np.sin(2*np.pi*np.cumsum(f)/sr)*0.4)*(t/d)**2.2*0.55
def boom(d=1.8):
    t=tt(d); f=70*np.exp(-t*1.2)+30; return (np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-t*2.2)+lp(rng.standard_normal(len(t)),300)*np.exp(-t*5)*0.6)*0.9
def crash(d=2.5):
    t=tt(d); return ff(rng.standard_normal(len(t)),lo=3500)*np.exp(-t*2.0)*0.35
def tick():
    t=tt(0.08); return ff(rng.standard_normal(len(t)),lo=3000)*np.exp(-t*70)*0.35
chords=[([220,261.63,329.63],55.0,[220,261.63,329.63,440]),([174.61,220,261.63],43.65,[174.61,220,261.63,349.23]),
        ([261.63,329.63,392],65.41,[261.63,329.63,392,523.25]),([196,246.94,293.66],49.0,[196,246.94,293.66,392])]
# pads full length
for b in range(int(TOT/2)+1): add(pad(chords[b%4][0],2.7),b*2.0,0,0.9 if b*2>=5 else 0.7)
# intro
add(riser(2.0),0,0,0.9); add(boom(),2.0); add(kick(),2.0,0,1.1); add(crash(),2.0,0.2,0.8)
for k in range(6): add(kick(),2.0+k*beat,0,0.5)
for k in range(12): add(hat(),2.0+k*beat/2,0.3,0.6)
add(riser(1.2),3.8,0,0.7)
for k in range(8): add(snare(),4.4+k*beat/4*0.8,0,0.25+k*0.05)
# groove 5.0..20.0 and drop 22..25
def groove(t0,t1,full=True):
    t=t0; step=0
    while t<t1-1e-6:
        bi=int((t-5.0)/2.0); c=chords[bi%4]; pos=((t-5.0)%2.0)/beat; bi_=int(round(pos))
        add(kick(),t,0,1.0)
        if bi_ in (1,3): add(snare(),t,0,0.9)
        add(bass(c[1],beat*0.9),t,0,1.0); add(hat(True),t+beat/2,0.3,0.5); add(hat(),t,0.3,0.5); add(tick(),t,0,0.5)
        for q in range(2): add(pluck(c[2][(step*2+q)%4]*2),t+q*beat/2,-0.3,0.9 if full else 0.6)
        step+=1; t+=beat
groove(5.0,20.0)
# chapter impacts
for tc in (5.0,8.0,11.0,14.0,17.0):
    add(whoosh(0.5),tc-0.5,0,0.8); add(boom(1.2),tc,0,0.55); add(crash(1.2),tc,0.3,0.4)
# finale build 19..20, rapid 20..22
add(riser(1.0),19.0,0,0.8)
for k in range(8): add(kick(),20.0+k*0.25,0,0.8); add(snare(),20.0+k*0.25+0.125,0,0.35)
add(riser(0.5),21.5,0,0.7)
# drop 22.0
add(boom(2.5),22.0,0,1.1); add(crash(3),22.0,0.2,1.0); add(kick(),22.0,0,1.2)
for c0 in (22.0,24.0):
    pass
groove(22.0,25.0)
add(whoosh(0.8),24.2,0,0.6)
# end
add(boom(2.5),25.0,0,0.9); add(crash(2.5),25.0,0.3,0.7); add(pad([220,261.63,329.63,440],2.2,0.3),25.0)
m=max(np.abs(L).max(),np.abs(R).max()); L/=m; R/=m
fi=int((TOT-1.2)*sr); fd=np.ones(N); fd[fi:]=np.linspace(1,0,N-fi)
L=np.tanh(L*0.85*fd*1.3)/np.tanh(1.3); R=np.tanh(R*0.85*fd*1.3)/np.tanh(1.3)
pcm=(np.stack([L,R],1)*32767).astype(np.int16)
w=wave.open(sys.argv[1],"wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes()); w.close()
