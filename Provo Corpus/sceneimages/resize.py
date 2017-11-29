import glob
import skimage.transform as skim 
from skimage.io import imread, imsave
from tqdm import tqdm
import sys
import os

#Tell us how many files are in the directory (to verfiy).
os.system('ls | wc -l')

#Get the images
all_images = []
exts = ['*.jpeg','*.JPG','*.gif','*.bmp','*.jpg']
for e in exts:
	all_images.extend(glob.glob(e))
all_images = ['/Users/rjafek/Desktop/projects/Provo Corpus/sceneimages/'+i for i in all_images]
print (len(all_images)) #This number should be a little bit smaller than total files.


#Reshape all the images.
for im_name in tqdm(all_images):
	img = imread(im_name)
	if img.shape != (600,800,3):
		print (im_name)
		img_out = skim.resize(img, (600,800,3), mode='constant')
		imsave(im_name, img_out)