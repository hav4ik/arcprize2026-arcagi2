# task 269e22fb pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 11 function programs)
def solve_orient(I):
    I=np.array(I); u=np.unique(I)
    H,W=orients[0].shape
    h,w=I.shape
    for oi,O in enumerate(orients):
        for a in range(2):
            c0,c1=(u[0],u[1]) if a==0 else (u[1],u[0])
            col=colorize(O,c0,c1)
            for i in range(H-h+1):
                for j in range(W-w+1):
                    if np.array_equal(col[i:i+h,j:j+w],I):
                        return col,(oi,i,j,a)
    return None,None

def colorize(O,c0,c1): return np.where(O==0,c0,c1)

# submitted:
answer,info=solve_orient(test_input) if False else (None,None)
answer=colorize(orients[oi],c0,c1)
answer = np.array(answer, dtype=int)
answer
