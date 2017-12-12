
import imageio
import glob
import os
from natsort import natsorted

def makegif(fld,outname, delete_files = False):
    # fld = '/Users/davesteps/Google Drive/pycharmProjects/keras_RL/model6_1bc90f81b393690023efa0cb0e4bb69935ee4b64/New Folder With Items'
    images = []
    filenames = natsorted(glob.glob(fld+'/*.png'))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(outname + '.gif', images)
    if delete_files:
        for filename in filenames:
            os.remove(filename)




