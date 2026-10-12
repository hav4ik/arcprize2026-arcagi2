# task 136b0064 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80 (median-length of 17 function programs)
def bands_columns(a):
    H=a.shape[0]
    L=[];R=[]
    r=0
    while r+2<H:
        L.append(a[r:r+3,0:3]); R.append(a[r:r+3,4:7]); r+=4
    return L,R

def solve(a):
    a=np.array(a)
    H=a.shape[0]
    L,R=bands_columns(a)
    seq=[int(g[g!=0][0]) for g in L]+[int(g[g!=0][0]) for g in R]
    # 5 position
    marker=np.argwhere(a==5)
    s=int(marker[0,1])-8
    out=np.zeros((H,7),dtype=int)
    out[0,s]=5
    last_row=0; col=s
    for c in seq:
        d,ln=rules[c]
        start=last_row+1
        if d=='D':
            for k in range(ln):
                out[start+k,col]=c
            last_row=start+ln-1
        elif d=='R':
            for k in range(ln):
                out[start,col+k]=c
            col=col+ln-1; last_row=start
        else:
            for k in range(ln):
                out[start,col-k]=c
            col=col-ln+1; last_row=start
    return out

# submitted:
solve(test_input)
