from pathlib import Path
import shutil, glob

def processFileName(name):
    nameArray = name.stem.split()
    g = 'F' if nameArray[1] == "Female" else 'M'
    num = nameArray[-1]
    return g + '_' + num

folderDir = Path("./3DScanStore/48RetopoFaces/")
for fileName in folderDir.glob('./[!.]*'):
    name = processFileName(fileName)
    try:
        objFolder = fileName.joinpath( "OBJ/Sub Division/")

        obj = objFolder.joinpath( "Head.OBJ")
        
        shutil.copy(obj, 'objs/' + name + '.obj')
    except:
        print("Error: " + fileName) 