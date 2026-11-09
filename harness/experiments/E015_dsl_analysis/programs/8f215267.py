# task 8f215267 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 18 function programs)
def solve(g):
    g=np.array(g)
    vals,counts=np.unique(g,return_counts=True)
    bg=int(vals[np.argmax(counts)])
    m=g!=bg
    lab,n=ndimage.label(m)
    frames=[]
    scattered={}
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        col=int(g[ys[0],xs[0]])
        H=ys.max()-ys.min()+1; W=xs.max()-xs.min()+1
        if H>=3 and W>=3 and len(ys)==2*H+2*W-4:
            frames.append((col,ys.min(),ys.max(),xs.min(),xs.max()))
        else:
            scattered.setdefault(col,0)
            scattered[col]+=1
    out=g.copy()
    out[m]=bg  # remove all non-bg
    # restore frames
    for col,y0,y1,x0,x1 in frames:
        out[y0:y1+1,x0]=col; out[y0:y1+1,x1]=col
        out[y0,x0:x1+1]=col; out[y1,x0:x1+1]=col
    # add marks
    for col,y0,y1,x0,x1 in frames:
        nmarks=scattered.get(col,0)
        W=x1-x0-1
        r=(y0+y1)//2
        for k in range(nmarks):
            c=x1-2-2*k
            out[r,c]=col
    return out

# submitted:
answer=solve(test_input)
answer
