# task 3dc255db pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub64a (median-length of 4 function programs)
def solve(grid):
    g=np.array(grid)
    H,W=g.shape
    # 8-connected components of nonzero
    seen=np.zeros_like(g,dtype=bool)
    out=g.copy()
    for i in range(H):
        for j in range(W):
            if g[i,j]!=0 and not seen[i,j]:
                # BFS
                comp=[]
                dq=deque([(i,j)]); seen[i,j]=True
                while dq:
                    x,y=dq.popleft(); comp.append((x,y))
                    for dx in(-1,0,1):
                        for dy in(-1,0,1):
                            if dx==0 and dy==0: continue
                            nx,ny=x+dx,y+dy
                            if 0<=nx<H and 0<=ny<W and g[nx,ny]!=0 and not seen[nx,ny]:
                                seen[nx,ny]=True; dq.append((nx,ny))
                colors={}
                for (x,y) in comp:
                    colors[g[x,y]]=colors.get(g[x,y],0)+1
                if len(colors)==1:
                    continue  # stray, leave
                container_color=max(colors,key=colors.get)
                content_color=[c for c in colors if c!=container_color][0]
                cont=[(x,y) for (x,y) in comp if g[x,y]==container_color]
                content=[(x,y) for (x,y) in comp if g[x,y]==content_color]
                contset=set(cont)
                # degrees
                deg={}
                for (x,y) in cont:
                    d=0
                    for dx in(-1,0,1):
                        for dy in(-1,0,1):
                            if dx==0 and dy==0: continue
                            if (x+dx,y+dy) in contset: d+=1
                    deg[(x,y)]=d
                endpoints=[c for c in cont if deg[c]<=1]
                cc=np.mean(np.array(cont),axis=0)
                if len(endpoints)==2:
                    ec=np.mean(np.array(endpoints),axis=0)
                    d=cc-ec
                    if abs(d[0])>=abs(d[1]):
                        tdir=(1 if d[0]>0 else -1,0)
                    else:
                        tdir=(0,1 if d[1]>0 else -1)
                    # tip cell extreme along tdir
                    dot=[c[0]*tdir[0]+c[1]*tdir[1] for c in cont]
                    tip=cont[int(np.argmax(dot))]
                elif len(endpoints)==3:
                    # branch point
                    branch=max(cont,key=lambda c:deg[c])
                    vecs=[]
                    for e in endpoints:
                        v=(e[0]-branch[0],e[1]-branch[1])
                        n=(v[0]**2+v[1]**2)**.5
                        vecs.append((e,(v[0]/n,v[1]/n)))
                    best=None
                    for k,(e,v) in enumerate(vecs):
                        s=[0.0,0.0]
                        for m,(e2,v2) in enumerate(vecs):
                            if m!=k: s[0]+=v2[0]; s[1]+=v2[1]
                        dotp=v[0]*s[0]+v[1]*s[1]
                        if best is None or dotp<best[0]:
                            best=(dotp,e,v)
                    tip=best[1]
                    v=best[2]
                    if abs(v[0])>=abs(v[1]):
                        tdir=(1 if v[0]>0 else -1,0)
                    else:
                        tdir=(0,1 if v[1]>0 else -1)
                else:
                    raise Exception("bad endpoints")
                # remove content
                for (x,y) in content: out[x,y]=0
                # place line
                cnt=len(content)
                x,y=tip[0]+tdir[0],tip[1]+tdir[1]
                placed=0
                while 0<=x<H and 0<=y<W and placed<cnt:
                    out[x,y]=content_color
                    x+=tdir[0]; y+=tdir[1]; placed+=1
    return out

# submitted:
answer=solve(test_input)
answer
