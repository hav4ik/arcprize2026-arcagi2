import gzip, zlib, glob, sys
def ratio(s):
    b=s.encode('utf-8','replace'); return len(zlib.compress(b,6))/max(len(b),1)
for d in sys.argv[1:]:
    files=glob.glob(d+'/*.gz'); n=0; hits=[]
    for f in files:
        t=gzip.open(f,'rt',errors='replace').read(); n+=1
        W=6000; worst=1.0
        for i in range(0,max(1,len(t)-W),W//2):
            worst=min(worst,ratio(t[i:i+W]))
        if worst<0.02: hits.append((f.split('/')[-1],round(worst,4)))
    print(d, 'files',n,'windows<0.02:',len(hits), hits[:8])
