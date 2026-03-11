import os
import os.path

rootdir = r'C:\software\mydemo\DEIM\datasets\Maize\test\labels/'
files = os.listdir(rootdir)

a=1

for name in files:
    print(name)
    newname = str(a) + '.txt'
    a = a+1
    os.rename(rootdir+name,rootdir+newname)
