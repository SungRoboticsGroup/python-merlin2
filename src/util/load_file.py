from numpy.typing import NDArray
import numpy as np
from copy import deepcopy
from ezdxf.filemanagement import readfile
import matplotlib.pyplot as plt
from typing import Tuple
import os
import xml.etree.ElementTree as ET

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
        fc = np.array(face)
        y_mins = np.where(vs[face, 1] == np.min(vs[face, 1]))[0]
        min_pt = y_mins[np.argmax(vs[fc[y_mins], 0])]

        a = vs[face[min_pt - 1 if min_pt > 0 else np.size(min_pt)]] - vs[face[min_pt]]
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

def load_dxf(filename : str, visualize : bool =False) -> Tuple[NDArray, NDArray]:
    """loads a dxf in as a planar unfolded pattern and returns the nodes and panels

    Parameters
    ----------
    filename : str
        The path to the dxf file
    visualize : bool, optional
        Shows a visualization of the pattern with numbered vertices once it's loaded. This can be useful for determining load and support conditions. By default False.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Returns (node, panel) in a format that can be passed into prepare_data

    Raises
    ------
    FileNotFoundError
        If the dxf file is not found
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist in the current directory {os.getcwd()}")

    doc = readfile(filename)
    msp = doc.modelspace()
    lines = []
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            lines += list(zip([list(i.dxf.start) for i in e.virtual_entities()], [list(i.dxf.end) for i in e.virtual_entities()])) # type: ignore
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

def load_svg(fp : str, visualize=False) -> Tuple[NDArray, NDArray]:
    """Loads an svg file as an unfolded planar pattern and returns the nodes and panels

    Parameters
    ----------
    fp : str
        The path to the svg file
    visualize : bool, optional
        Shows a visualization of the svg with numbered vertices after loading it. This is useful for determining supp and load conditions. By default False.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Returns a tuple of (node, panel), each of which can be passed directly into prepare_data

    Raises
    ------
    FileNotFoundError
        If the svg file is not found.
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} does not exist in the current directory {os.getcwd()}")
    
    lines = []
    tree = ET.parse(fp)
    root = tree.getroot()
    for line in root:
        if "line" in line.tag:
            x1 = line.get("x1")
            x2 = line.get("x2")
            y1 = line.get("y1")
            y2 = line.get("y2")
            if x1 is None or x2 is None or y1 is None or y2 is None:
                continue
            lines.append(([float(x1), float(y1), 0.0], [float(x2), float(y2), 0.0]))
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


def load_obj(fp : str) -> Tuple[NDArray, NDArray]:
    """Loads an obj file into the simulation

    Parameters
    ----------
    fp : str
        The path to the obj file

    Returns
    -------
    Tuple[NDArray, NDArray]
        Returns a tuple containing (nodes, panels) in a format that can be passed directly into prepare_data

    Raises
    ------
    FileNotFoundError
        If the obj file is not found
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} does not exist in the current directory {os.getcwd()}")

    with open(fp, "r") as f:
        node = []
        panel = []

        while True:
            tline = f.readline()
            if tline == "":
                break
            tline = tline[: -1]
            ln = tline.split()[0]
            if ln == "v":
                node.append([f for f in tline.split(" ")[1 : ]])
            elif ln == "f":
                line = tline[2 : ]

                allind = line.replace("/", " ").split()
                nf = line.count("/", 0, line.index(" ")) + 1
                panel.append([int(f) - 1 for f in allind[ : : nf]])

    return np.array(node, dtype=float), np.array(panel, dtype=object)


if __name__ == "__main__":
    # vs = np.array([[0, 0], [9, 5], [13, 10], [16, 4], [10, -5]])
    # adj = [[1, 4], [0, 2, 3, 4], [1, 3], [1, 2, 4], [0, 1, 3]]
    # f = _embedding_to_faces({"vertices":vs, "adjacencies":adj})
    vs, fcs = load_dxf("sq2.dxf", True)
    plt.show()
    pass