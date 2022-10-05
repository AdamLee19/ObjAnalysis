import os, shutil, glob
def processFileName(name):
    nameArray = name.split()
    g = 'F' if nameArray[1] == "Female" else 'M'
    num = nameArray[-1]
    return g + '_' + num

folderDir = "./3DScanStore/48RetopoFaces/"
for fileName in os.listdir(folderDir):
    if fileName == '.DS_Store': continue
    name = processFileName(fileName)
    try:
        objFolder = os.path.join(folderDir, fileName, "OBJ/Sub Division/")
        objs = glob.iglob(os.path.join(objFolder, "Head.OBJ"))
        for obj in objs:
            shutil.copy(obj, 'objs/' + name + '.obj')
    except:
        print("Error: " + fileName) 