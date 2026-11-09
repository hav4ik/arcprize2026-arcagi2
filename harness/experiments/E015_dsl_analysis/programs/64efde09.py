# task 64efde09 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40 (median-length of 5 function programs)
def startend(l):
    # end0 if l[0]==l[1], else end -1
    if l[0]==l[1]: return 0
    return len(l)-1

def maxrun(l):
    best=1;cur=1
    for i in range(1,len(l)):
        if l[i]==l[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    return best

def solve(grid):
    inp=np.array(grid); H,W=inp.shape
    nz=(inp!=8).astype(int)
    lab,n=ndimage.label(nz, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    shapes=[]; markers=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        if len(ys)==1: markers.append((ys[0],xs[0],inp[ys[0],xs[0]]))
        else: shapes.append((ys.min(),ys.max(),xs.min(),xs.max()))
    info=[]
    for (y0,y1,x0,x1) in shapes:
        h=y1-y0+1; w=x1-x0+1
        if h>w:
            lines=[inp[y0:y1+1,x0], inp[y0:y1+1,x1]]
            rl=[maxrun(l) for l in lines]
            rail=0 if rl[0]>=rl[1] else 1; pattern=1-rail
            start=startend(lines[pattern])
            info.append(dict(orient='v',y0=y0,y1=y1,x0=x0,x1=x1,L=h,rail=rail,start=start))
        else:
            lines=[inp[y0,x0:x1+1], inp[y1,x0:x1+1]]
            rl=[maxrun(l) for l in lines]
            rail=0 if rl[0]>=rl[1] else 1; pattern=1-rail
            start=startend(lines[pattern])
            info.append(dict(orient='h',y0=y0,y1=y1,x0=x0,x1=x1,L=w,rail=rail,start=start))
    colors={}
    for (my,mx,mc) in markers:
        for sh in info:
            if sh['orient']=='v':
                if sh['y0']<=my<=sh['y1']:
                    if sh['rail']==0 and mx<sh['x0']: side=True
                    elif sh['rail']==1 and mx>sh['x1']: side=True
                    else: side=False
                    if side:
                        rel=my-sh['y0']; k=rel if sh['start']==0 else sh['L']-1-rel
                        colors[k]=mc; break
            else:
                if sh['x0']<=mx<=sh['x1']:
                    if sh['rail']==0 and my<sh['y0']: side=True
                    elif sh['rail']==1 and my>sh['y1']: side=True
                    else: side=False
                    if side:
                        rel=mx-sh['x0']; k=rel if sh['start']==0 else sh['L']-1-rel
                        colors[k]=mc; break
    out=inp.copy()
    for sh in info:
        for k,c in colors.items():
            rel=k if sh['start']==0 else sh['L']-1-k
            if sh['orient']=='v':
                r=sh['y0']+rel
                if sh['rail']==0:
                    for x in range(sh['x0']-1,-1,-1):
                        if inp[r,x]!=8: break
                        out[r,x]=c
                else:
                    for x in range(sh['x1']+1,W):
                        if inp[r,x]!=8: break
                        out[r,x]=c
            else:
                col=sh['x0']+rel
                if sh['rail']==0:
                    for y in range(sh['y0']-1,-1,-1):
                        if inp[y,col]!=8: break
                        out[y,col]=c
                else:
                    for y in range(sh['y1']+1,H):
                        if inp[y,col]!=8: break
                        out[y,col]=c
    return out

# submitted:
answer=solve(test_input)
answer
