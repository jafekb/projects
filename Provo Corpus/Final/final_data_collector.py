import glob
from tqdm import tqdm
import cv2
import os
import sys
from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd

def get_RMS_contrast(arr):
	#Source: https://en.wikipedia.org/wiki/Contrast_(vision)
	m,n = np.shape(arr)
	avg = arr.mean()
	rms = np.sqrt(np.sum((arr-avg)**2)/(m*n))
	return rms

def get_brightness(arr):
	"""
	Brightness calculated using a Luma channel
	https://en.wikipedia.org/wiki/Luma_(video)
	"""
	R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
	Y = 0.299*R + 0.587*G + 0.144*B
	return Y.mean()

def spit_out_CSV(how_long, everything):
	dfout = pd.DataFrame(everything)
	dfout = dfout.transpose()
	dfout.to_csv(how_long+'_image_info.csv')

def main(im_names):
	everything = {}
	print (len(im_names))
	cropper_length = 'Croppers/' #This will be helpful later in your journey.

	for im_name in tqdm(im_names):
		#This is how we will store the information about this particular image
		one_img = {}

		#Format the image name
		name = os.path.basename(im_name)
		splits = name.split('_')
		if len(splits)==3:
			name, x, y = splits
		elif len(splits)==4:
			name1, name2, x, y = splits
			name = '_'.join([name1, name2])
		else:
			print (im_name)
			raise ValueError('whats up?')
		y = y.split('.')[0] #take off the '.png' part at the end.
		#Also keep in mind that this is upside-down.


		#Get the images we need (color, B&W, vertical, horizontal)
		color_img = plt.imread(im_name).astype(np.float64)
		img = color_img.mean(axis=2) #B&W
		tot_pixels = color_img.sum()

		vert_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
		horiz_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

		#Get image info
		bright = get_brightness(color_img)
		contr = get_RMS_contrast(img)
		fm = cv2.Laplacian(img, cv2.CV_64F).var()
		vert_contr = get_RMS_contrast(vert_img)
		vert_fm = cv2.Laplacian(vert_img, cv2.CV_64F).var()
		horiz_contr = get_RMS_contrast(horiz_img)
		horiz_fm = cv2.Laplacian(horiz_img, cv2.CV_64F).var()
		#Colors
		R, G, B = color_img[:,:,0], color_img[:,:,1], color_img[:,:,2]
		if tot_pixels > 0:
			red_prop = R.sum()/tot_pixels
			green_prop = G.sum()/tot_pixels
			blue_prop = B.sum()/tot_pixels
		else:
			red_prop, green_prop, blue_prop = 0, 0, 0

		#Put all that information in the DataFrame
		one_img['Brightness'] = bright
		one_img['Contrast'] = contr 
		one_img['FM'] = fm 
		one_img['VertContrast'] = vert_contr
		one_img['VertFM'] = vert_fm
		one_img['HorizContrast'] = horiz_contr
		one_img['HorizFM'] = horiz_fm
		one_img['Red'] = red_prop
		one_img['Green'] = green_prop
		one_img['Blue'] = blue_prop
		one_img['X'] = x 
		one_img['Y'] = y

		#Save that information to a CSV
		everything[im_name[len(cropper_length):]] = one_img
		
		# spit_out_CSV('PARTIAL', everything) 
		#^Putting in the above line allows us to update the CSV each iteration so we don't lose our progess, 
		#but takes a long time each iteration, especially as the dictionary gets big.

	spit_out_CSV('FULL', everything)


if __name__ == '__main__':
	im_names = glob.glob('Croppers/*')
	main(im_names)