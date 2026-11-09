# task e12f9a14 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 1 function programs)


# submitted:
import numpy as np
g=np.array(test_input).copy()
rays=[]
# color 2
rays += [(2,[(6,22),(5,21),(4,20)]+[(r,20) for r in range(3,-1,-1)]),
         (2,[(6,25),(5,26),(4,27),(3,28),(2,29)]),
         (2,[(9,25),(10,26),(11,27),(12,28),(13,29)]),
         (3,[(10,4),(9,3),(8,2),(7,1),(6,0)]),
         (3,[(10,7),(9,8)]+[(r,8) for r in range(8,-1,-1)]),
         (3,[(r,5) for r in range(13,30)]),
         (3,[(13,7),(14,8)]+[(r,8) for r in range(15,30)]),
         (4,[(10,10),(9,9)]+[(r,9) for r in range(8,-1,-1)]),
         (4,[(10,13),(9,14),(8,15),(7,16),(6,17),(5,18),(4,19)]+[(r,19) for r in range(3,-1,-1)]),
         (4,[(13,10),(14,9)]+[(r,9) for r in range(15,30)]),
         (4,[(r,12) for r in range(13,30)]),
         (7,[(24,26),(23,27),(22,28),(21,29)]),
         (7,[(26,26),(26,27),(26,28),(26,29)]),
         (7,[(27,24),(28,24),(29,24)])]
for col,cells in rays:
    for (r,c) in cells:
        if 0<=r<g.shape[0] and 0<=c<g.shape[1]:
            g[r,c]=col
answer=g
for row in answer:
    print(''.join('.' if v==0 else str(v) for v in row))
answer
