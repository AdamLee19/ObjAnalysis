from numpy import load, save



def readFile(dir, mode = 'rb'):
    with open(dir, mode) as f:
        matrix = load(f)
    
    return matrix


def writeFile(dir, npArray,  mode = 'wb'):
    with open(dir, mode) as f:
        save(f, npArray)