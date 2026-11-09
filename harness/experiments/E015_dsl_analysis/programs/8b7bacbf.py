# task 8b7bacbf pair 1 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 4 function programs)
def detect_holes(a):
    H,W=a.shape
    outside=np.zeros_like(a,dtype=bool); dq=deque()
    for (r,c) in [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]:
        if a[r,c]==0: outside[r,c]=True; dq.append((r,c))
    while dq:
        r,c=dq.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not outside[nr,nc] and a[nr,nc]==0:
                outside[nr,nc]=True; dq.append((nr,nc))
    return ndimage.label((a==0)&(~outside))[0]

def solve_full(a):
    a=np.array(a); H,W=a.shape
    out=a.copy()
    cnt=Counter(a.flatten().tolist())
    markers=[c for c,k in cnt.items() if k==1 and c!=0]
    labh=detect_holes(a); nh=labh.max()
    rings={}; ringmask=np.zeros((H,W),bool)
    for h in range(1,nh+1):
        cells=list(zip(*[x.tolist() for x in np.where(labh==h)]))
        ring=set()
        for (r,c) in cells:
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and a[nr,nc]!=0: ring.add((nr,nc))
        rings[h]=ring
        for rc in ring: ringmask[rc]=True
    valid={h:(len(set(int(a[r,c]) for r,c in rings[h]))==1) for h in rings}
    wire=(a!=0)&(~ringmask)
    labw,_=ndimage.label(wire,structure=st)
    from collections import defaultdict
    hw=defaultdict(set)
    for h in range(1,nh+1):
        ws=set()
        for (r,c) in rings[h]:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    nr,nc=r+dr,c+dc
                    if 0<=nr<H and 0<=nc<W and labw[nr,nc]>0: ws.add(labw[nr,nc])
        hw[h]=ws
    for sp in markers:
        mr,mc=np.argwhere(a==sp)[0]
        start=set()
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                nr,nc=mr+dr,mc+dc
                if 0<=nr<H and 0<=nc<W and labw[nr,nc]>0: start.add(labw[nr,nc])
        wiresdone=set(start); holesdone=set(); wq=deque(start)
        while wq:
            w=wq.popleft()
            for h in range(1,nh+1):
                if w in hw[h] and h not in holesdone:
                    holesdone.add(h)
                    for w2 in hw[h]:
                        if w2 not in wiresdone: wiresdone.add(w2); wq.append(w2)
        for h in holesdone:
            if valid[h]: out[labh==h]=sp
    return out

# submitted:
answer=solve_full(test_input)
answer
