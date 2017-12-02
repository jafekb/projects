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
csv2 = ['attic.jpg','basketballcourt5.jpg', 'butchers_shop.jpg', 'carnival.jpg',
				'cavern2.jpg', 'closet.jpg', 'computerlab4.jpg', 'creek.jpg',
				'desert.jpg', 'football_stadium.jpg', 'HELIPAD.jpg',
				'herb_garden.jpg', 'hospitalward.jpg', 'lockerroom4.jpg',
				'marsh1.jpg', 'moun_tains.jpg', 'outdoortheme2.jpg', 
				'quarry2.jpg', 'Volleyball_Outdoor.jpg', 'zoocage2.jpg']
csvs = ['computerlab4.jpg', 'Warehouse1.jpg', 'diningroom.jpg', 'forest1.jpg',
			'FOUNTAIN.jpg', 'generalstore.jpg', 'ocean2.jpg', 'playroom.jpg',
			'pool_hall.JPG', 'quarry.jpg', 'Rock_Arch.jpg', 'slum2.jpg',
			'video_store.jpg']
im_names = csv2+csvs
im_names = ['/home/jafekb@byu.local/myacmeshare/projects/Provo Corpus/sceneimages/'+i for i in im_names]
everything = {}
count = 0

print(len(im_names))

for im_name in im_names:
	print (os.path.basename(im_name))
	
	image = plt.imread(im_name)

	for ROW in tqdm(range(0, 600, 5)):
		for COL in range(0, 800, 5):
			#Pixel identification information
			one_row = {}
			count += 1

			#Get the circle image.
			color_localized, nonblack = circle(ROW, COL, image)

			#Save the identification information
			tot = color_localized.sum()
			one_row['Total'] = tot
			one_row['picture'] = os.path.basename(im_name)
			one_row['X_center'] = ROW
			one_row['Y_center'] = COL

			#RGB
			R, G, B = color_localized[:,:,0], color_localized[:,:,1], color_localized[:,:,2]
			if tot > 0:
				red_proportion = R.sum()/tot
				green_proportion = G.sum()/tot
				blue_proportion = B.sum()/tot
			else:
				red_proportion, green_proportion, blue_proportion = 0, 0, 0
			
			one_row['Red'] = red_proportion
			one_row['Green'] = green_proportion
			one_row['Blue'] = blue_proportion

			everything[count] = one_row

			#Spit out lines to a CSV
			dfout = pd.DataFrame(everything)
			dfout = dfout.transpose()
			dfout.to_csv('ColorProportions/{}.csv'.format(os.path.basename(im_name).split('.')[0]), index=False)





