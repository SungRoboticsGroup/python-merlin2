import numpy as np

def write_to_obj(filename, node, trigl, bars=None, bend = None, description = "This is an output from MERLIN2"):
    #specify viewpoint to unify vertex winding
    vp = np.hstack((np.max(node[:, 0]) / 2, np.max(node[:, 1]) / 2, 10 * (np.max(node[:, 2]) + 1)))

    with open(filename, "w") as fid:
        fid.write(f"# {description}\n")

        # write vertex data
        for i in range(np.size(node, 0)):
            n = node[i].astype(str)
            fid.write(f"v {' '.join(n)}\n")
        
        fid.write(f"# {np.size(node, 0)} vertices\n")

        for i in range(np.size(trigl, 0)):
            a = node[trigl[i, 1]] - node[trigl[i, 0]]
            b = node[trigl[i, 2]] - node[trigl[i, 1]]
            n = (np.sum(node[trigl[i]], 0) / 3) - vp
            if (np.cross(a, b) @ n) < 0:
                trigli = trigl[i, :]
            else:
                trigli = np.flip(trigl[i])
            fid.write(f"f {' '.join((trigli + 1).astype(str))}\n")
        
        fid.write(f"# {np.size(trigl, 0)} triangles\n")

        if bars is not None and bend is not None:
            bars += 1
            sz_bend = np.size(bend, 0)
            for i in range(sz_bend):
                fid.write(f"#e {bars[i][0]} {bars[i][1]} 1\n")
            fid.write(f"# {sz_bend} bending lines\n")
            for i in range(sz_bend, np.size(bars, 0)):
                fid.write(f"#e {bars[i][0]} {bars[i][1]} 0\n")
            fid.write(f"# {np.size(bars, 0) - sz_bend} generic (folding/boundary) edges\n")
    


