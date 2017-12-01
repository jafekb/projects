from __future__ import division, print_function
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import sys
import glob
from tqdm import tqdm
import os

def circle(center_x, center_y, img):
	r = 75 #This relates pretty well to how much a human sees.
	count = 0
	l = img.shape
	m, n = l[0], l[1] #It also might have a depth dimension
	arr = np.zeros_like(img)

	for i in range(m): #I couldn't figure out how to mask this as a circle...
		for j in range(n):
			if (center_x - j)**2 + (center_y - i)**2 <= r**2:
				count +=1
				arr[i,j] = img[i,j]

	return arr, count

n = 20
exts = ['.jpeg', '.JPG', '.gif', '.bmp', '.jpg']
im_names = []
for ext in exts:
	im_names.extend(glob.glob('/Users/rjafek/Desktop/projects/Provo Corpus/sceneimages/*'+ext))

colors = ['Red', 'Green', 'Blue']

everything = {}
count = 0

for im_name in im_names:
	one_row = {}
	image = plt.imread(im_name)

	for ROW in tqdm(range(0, 600, 5)):
		for COL in range(0, 800, 5):
			count += 1
			color_localized, nonblack = circle(ROW, COL, image)

			tot = color_localized.sum()
			one_row['Total'] = tot
			one_row['picture'] = os.path.basename(im_name)
			one_row['X_center'] = ROW
			one_row['Y_center'] = COL

			for j in range(3):
				copy = np.copy(color_localized)
				channel = copy[:,:,j]
				wrong = [k for k in range(3) if k!=j]
				for n in wrong:
					copy[:,:,n] = 0

				one_row[colors[j]] = copy.sum()/tot

			everything[count] = one_row

			#Spit out lines to a CSV
			dfout = pd.DataFrame(everything)
			dfout = dfout.transpose()
			dfout.to_csv('Color_Proportions.csv', index=False)

			if count ==5:
				sys.exit(0)




