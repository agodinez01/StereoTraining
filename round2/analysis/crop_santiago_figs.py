from PIL import Image
import os
from os import listdir
from os.path import isfile, join

folders = ['DEF_24sept_expFit_StartEndMean_40 blocks/', 'DEF_24sept_expFit_threeConditions_Means_40 blocks/', 'rev2020.10.02/', 'DEF_23sept_arrows/', 'new_figs/']

for folder in folders:
    myPath = r'C:/Users/angie/Box/Projects/1.RecoverStereoVR/round2/image_analysis/' + folder
    os.chdir(myPath)

    files = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    cropped = '_cropped.tif'

    for file in files:
        try:
            imgPath = myPath + file
            if imgPath.endswith(cropped):
                print('File exist')

            else:
                img = Image.open(imgPath)
                box = (175, 0, 1575, 1313)
                croppedImage = img.crop(box)
                croppedImage.save(file[:-4] + '_cropped.tif')

        except FileNotFoundError:
            print('Image path is not found')