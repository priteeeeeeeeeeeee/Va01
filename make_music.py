import numpy as np, wave
sr=44100; BPM=120; beat=60/BPM; TOT=18.8
N=int(TOT*sr); L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(3)
def add(sig,t,pan=0.0,gain=1.0):
    i=int(t*sr)
    if i>=N: return
    s=sig[:N-i]
    L[i:i+len(s)]+=s*gain*(1-max(pan,0)); R[i:i+len(s)]+=s*gain*(1+min(pan,0))
def tt(d): return np.arange(int(d*sr))/sr
def fft_filter(x,lo=None,hi=None):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr)
    m=np.ones_like(f)
    if lo: m*= (f>=lo)
    if hi: m*= (f<=hi)
    return np.fft.irfft(X*m,len(x))
def lp(x,fc):  # smooth lowpass
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr)
    return np.fft.irfft(X/(1+(f/fc)**4),len(x))
def kick():
    t=tt(0.35); f=45+110*np.exp(-t*28); ph=2*np.pi*np.cumsum(f)/sr
    return np.sin(ph)*np.exp(-t*9)+0.3*np.exp(-t*80)*rng.standard_normal(len(t))*0.2
def snare():
    t=tt(0.25); n=rng.standard_normal(len(t)); n=fft_filter(n,lo=1500)
    return (n*np.exp(-t*20)*0.5+np.sin(2*np.pi*190*t)*np.exp(-t*25)*0.4)
def hat(open_=False):
    t=tt(0.12 if not open_ else 0.3); n=fft_filter(rng.standard_normal(len(t)),lo=7000)
    return n*np.exp(-t*(60 if not open_ else 14))*0.25
def bass(f,d):
    t=tt(d); s=np.sin(2*np.pi*f*t)+0.3*np.sin(4*np.pi*f*t)
    return lp(s,400)*np.minimum(1,t*80)*np.minimum(1,(d-t)*40)*0.6
def pluck(f,d=0.4):
    t=tt(d); s=sum(np.sin(2*np.pi*f*k*t)/k for k in (1,2,3))
    return s*np.exp(-t*9)*0.18
def pad(freqs,d):
    t=tt(d); s=np.zeros(len(t))
    for f in freqs:
        for det in (-0.4,0,0.4):
            s+=np.sign(np.sin(2*np.pi*(f+det)*t))*0.5+np.sin(2*np.pi*(f+det)*t)
    s=lp(s,900)/len(freqs)/3
    env=np.minimum(1,t/0.6)*np.minimum(1,(d-t)/0.6)
    return s*env*0.22
def whoosh(d=0.5,rise=True):
    t=tt(d); n=rng.standard_normal(len(t))
    # band sweep via amplitude-modulated filtered noise (two bands crossfaded)
    b=fft_filter(n,lo=800,hi=6000)
    env=(t/d)**2 if rise else (1-t/d)**2
    return b*env*0.5*np.minimum(1,(d-t)*30)
def riser(d):
    t=tt(d); n=fft_filter(rng.standard_normal(len(t)),lo=500)
    f=200+2400*(t/d)**2; tone=np.sin(2*np.pi*np.cumsum(f)/sr)
    return (n*0.5+tone*0.4)*(t/d)**2.2*0.5
def boom():
    t=tt(1.6); f=70*np.exp(-t*1.2)+30; ph=2*np.pi*np.cumsum(f)/sr
    n=lp(rng.standard_normal(len(t)),300)*np.exp(-t*5)*0.6
    return (np.sin(ph)*np.exp(-t*2.2)+n)*0.9
def shimmer(d=2.5):
    t=tt(d); n=fft_filter(rng.standard_normal(len(t)),lo=5000)
    return n*np.exp(-t*2.2)*0.18
def tick():
    t=tt(0.08); return fft_filter(rng.standard_normal(len(t)),lo=3000)*np.exp(-t*70)*0.35
# chords per bar (2s): Am F C G ; notes
chords=[([220,261.63,329.63],55.0,[220,261.63,329.63,440]),
        ([174.61,220,261.63],43.65,[174.61,220,261.63,349.23]),
        ([261.63,329.63,392],65.41,[261.63,329.63,392,523.25]),
        ([196,246.94,293.66],49.0,[196,246.94,293.66,392])]
bars=int(TOT/(4*beat))+1
# intro pad 0-4 (bars 0,1) softer; beat section from 4.0 to 17.0
for b in range(bars):
    t0=b*4*beat; c=chords[b%4]
    add(pad(c[0],4*beat+0.5),t0,0,1.0 if t0>=4 else 0.8)
# intro riser into title hit at 2.0, shimmer
add(riser(2.0),0.0,0,0.9)
add(boom(),2.0,0,1.0); add(kick(),2.0,0,1.1); add(shimmer(),2.0,0.3,1)
# intro soft pulse: kicks light on beats 2.0..4.0 with hats
for k in range(4): add(kick(),2.0+k*beat,0,0.5); 
for k in range(8): add(hat(),2.0+k*beat/2,0.3,0.6)
# whoosh into 3.0 (top/bottom text) 
add(whoosh(0.4),2.7,0,0.5)
# build into drop at 4.0
add(riser(1.0),3.0,0,0.6)
for k in range(8): add(snare(),3.5+k*beat/4,0,0.25+k*0.05)
# main groove 4.0 -> 17.0
t=4.0; end=17.0; step=0; 
while t<end-1e-6:
    bi=int((t-4.0)/(4*beat)); c=chords[bi%4]
    pos=(t-4.0)%(4*beat)
    beat_i=round(pos/beat) if abs(pos/beat-round(pos/beat))<1e-6 else None
    if beat_i is not None:
        add(kick(),t,0,1.0)
        if beat_i in (1,3): add(snare(),t,0,0.9)
        add(bass(c[1],beat*0.9),t,0,1.0)
        add(hat(True),t+beat/2,0.3,0.5)
        add(tick(),t,0,0.7)   # cut tick on every beat cut accent
    add(hat(),t,0.3,0.5) if beat_i is not None else None
    # 16th arp
    for q in range(2):
        tn=t+q*beat/2
        add(pluck(c[2][(step*2+q)%4]*2),tn,-0.3,0.9)
    step+=1; t+=beat
# whoosh into each cut accent: put a short reverse whoosh before every 2nd beat
t=4.0
while t<end:
    add(whoosh(0.28),t-0.28,0,0.6); t+=beat*2
# ending: riser, impact at 17.0, outro pad
add(riser(1.0),16.0,0,0.5)
add(boom(),17.0,0,0.9); add(shimmer(),17.0,0.3,1)
add(pad([220,261.63,329.63],2.5),17.0,0,1.0)
# master: gentle fade out, normalize
m=np.maximum(np.abs(L).max(),np.abs(R).max()); L/=m; R/=m
fade=np.ones(N); fi=int((TOT-1.4)*sr); fade[fi:]=np.linspace(1,0,N-fi)
L*=fade*0.85; R*=fade*0.85
# soft clip
L=np.tanh(L*1.3)/np.tanh(1.3); R=np.tanh(R*1.3)/np.tanh(1.3)
pcm=(np.stack([L,R],1)*32767).astype(np.int16)
w=wave.open("music.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes()); w.close()
