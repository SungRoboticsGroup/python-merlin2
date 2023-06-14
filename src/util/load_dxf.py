from numpy.typing import NDArray
import numpy as np
from copy import deepcopy
from ezdxf.filemanagement import readfile
import matplotlib.pyplot as plt
from typing import Tuple

def _vector_angle(a : NDArray, b : NDArray):
    """a, b are the endpoints of the vector"""
    v1 = b - a  
    ang = np.arctan2(v1[1], v1[0])
    if ang < 0:
        ang += 2 * np.pi
    return ang
    


def _sort_adjacencies(vertices, adj, order="CCW"):
    """Sorts the adjacency list by angle"""
    for i in range(len(vertices)):
        adj[i] = sorted(adj[i], key = lambda x: _vector_angle(vertices[i], vertices[x]), reverse = (order == "CW"))
    return adj

def _next_edge(v_from : int | None, c_adj : list, c_adj_full : list):
    if v_from == None:
        return 0
    c = c_adj_full.index(v_from) + 1
    while c >= len(c_adj_full) or c_adj_full[c] not in c_adj:
        if c >= len(c_adj_full) - 1:
            c = 0
        else:
            c = c + 1
    return c_adj.index(c_adj_full[c])
    


def _embedding_to_faces(g) -> NDArray:
    """Gets the faces from a connected planar graph G = (V, E) given an adjacency list"""
    vs = g["vertices"]
    adj = _sort_adjacencies(vs, g["adjacencies"])
    a_old = deepcopy(adj)
    faces = []

    # traverses the faces in clockwise order
    while True:
        try:
            curr = next(x for x in range(len(adj)) if len(adj[x]) > 0)
        except:
            break
        start = curr
        face = [start]
        first = True
        fr = None
        while np.any(vs[curr] != vs[start]) or first:
            first = False
            old = curr
            curr = adj[curr][(_next_edge(fr, adj[curr], a_old[curr]))]
            if fr != None:
                adj[fr].remove(old)
            face.append(curr)
            fr = old
        adj[fr].remove(curr)
        faces.append(face)

    # remove the only oppositely-oriented face
    ccw = []
    cw = []
    for face in faces:
        # find the bottom right vertex
        y_mins = np.where(vs[face, 1] == np.min(vs[face, 1]))[0]
        min_pt = y_mins[np.argmax(vs[np.where(vs[face, 1] == np.min(vs[face, 1])), 0])]

        a = vs[face[min_pt - 1 if min_pt > 0 else np.size(min_pt, 1)]] - vs[face[min_pt]]
        b = vs[face[min_pt + 1 if min_pt < np.size(face) - 1 else 0]] - vs[face[min_pt]]
        if np.sign(np.cross(a, b)) < 0:
            cw.append(face[:-1])
        else:
            ccw.append(face[:-1])
    if len(ccw) <= 1:
        return np.array(cw, dtype=object)
    elif len(cw) <= 1:
        return np.array(ccw, dtype=object)
    else:
        raise ValueError("Neither list of faces can be clearly eliminated.")

def _round(v, p=0):
    return round(v[0], p), round(v[1], p), round(v[2], p)

def _convert_lines_to_graph(lines : list):
    """Each line is passed as [(x0, y0, z0), (x1, y1, z1)]"""
    point_idxs = {}
    vertices = []
    c = 0
    lines = [[_round(i[0], 2),_round(i[1], 2)] for i in lines]
    # First assign each start point and end point a unique number. These numbers will be considered vertex numbers.
    for line in lines:
        if point_idxs.get(line[0]) == None:
            point_idxs[line[0]] = c
            c += 1
            vertices.append(line[0])

        if point_idxs.get(line[1]) == None:
            point_idxs[line[1]] = c
            c += 1
            vertices.append(line[1])
    

    adj = [set() for _ in vertices]
    # Now create an adjacency list
    for line in lines:
        # add a bidirectional edge
        adj[point_idxs[line[0]]].add(point_idxs[line[1]])
        adj[point_idxs[line[1]]].add(point_idxs[line[0]])

    return {
        "vertices": np.array(vertices)[:, :2], 
        "adjacencies": adj
            }

def load_dxf(filename, visualize=False) -> Tuple[NDArray, NDArray]:
    """loads a dxf in as a planar unfolded pattern"""
    doc = readfile(filename)
    msp = doc.modelspace()
    lines = []
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            lines += list(zip([i.dxf.start for i in e.virtual_entities()], [i.dxf.end for i in e.virtual_entities()])) # type: ignore
    graph = _convert_lines_to_graph(lines)
    if visualize:
        for i in range(len(graph["adjacencies"])):     
            for j in graph["adjacencies"][i]: 
                coords = np.vstack((graph["vertices"][i], graph["vertices"][j]))
                plt.plot(coords[:, 0], coords[:, 1])
        for i in range(len(graph["vertices"])):
            plt.text(graph["vertices"][i][0], graph["vertices"][i][1], str(i))
        plt.show()
    return np.hstack((graph["vertices"], np.zeros((np.size(graph["vertices"], 0), 1)))), _embedding_to_faces(graph)

    

if __name__ == "__main__":
    # vs = np.array([[0, 0], [9, 5], [13, 10], [16, 4], [10, -5]])
    # adj = [[1, 4], [0, 2, 3, 4], [1, 3], [1, 2, 4], [0, 1, 3]]
    # f = _embedding_to_faces({"vertices":vs, "adjacencies":adj})
    vs, fcs = load_dxf("sq2.dxf", True)
    plt.show()
    pass