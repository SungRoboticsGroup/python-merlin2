import numpy as np
import os


def load_obj(fp : str):
    if not os.path.exists(fp):
        raise ValueError(f"File {fp} does not exist in the current directory {os.getcwd()}")
    
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
    
    return np.array(node, dtype=np.longdouble), np.array(panel, dtype=object)

if __name__ == "__main__":
    node, panel = load_obj("example/GMiura_FreeformOri.obj")
    print(node)
    print()
    print()
    print(panel)