
import imageio
import glob

def makegif(fld,outname):
    images = []
    filenames = glob.glob(fld+'/*.png')
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(outname + '.gif', images)
