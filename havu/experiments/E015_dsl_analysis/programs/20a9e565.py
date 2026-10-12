# task 20a9e565 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 4 function programs)
def gen_block(b):
    Lall=12-2*(b-1); Rall=16+2*(b-1); W=Rall-Lall+1
    g=np.zeros((2*b,W),dtype=int)
    def color(r): return 3 if (r+b)%2==0 else 2
    for r in range(1,b+1):
        L=12-2*(r-1)-Lall; R=16+2*(r-1)-Lall
        A=2*(r-1); B=A+1; c=color(r)
        if r==1:
            g[A,L:R+1]=c; g[B,L]=c; g[B,R]=c
        else:
            g[A,L:L+3]=c; g[A,R-2:R+1]=c
            ci=color(r-1); g[B,L]=c; g[B,R]=c; g[B,L+2]=ci; g[B,R-2]=ci
    return g

# submitted:
answer=gen_block(4)
answer
