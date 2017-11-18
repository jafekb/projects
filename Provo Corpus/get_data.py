import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt 
import cv2
from tqdm import tqdm
import os

def get_RMS_contrast(arr):
	m,n = np.shape(arr)
	rms = np.sqrt(np.sum((arr-arr.mean())**2)/(m*n))
	return rms

def get_brightness(arr):
	"""
	Brightness calculated using a Luma channel
	https://en.wikipedia.org/wiki/Luma_(video)
	"""
	R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
	Y = 0.299*R + 0.587*G + 0.144*B
	return Y.mean()

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

def main(df, overwrite=True):
	all_pics = list(set(df['picture']))
	for pic_name in all_pics:
		if overwrite:
		with open('{}.csv'.format(pic_name.split('.')[0]), 'w') as f:
			f.write('Picture,X_center,Y_center,')
			f.write('Brightness,Contrast,FM,')
			f.write('VerticalContrast,VerticalFM,',)
			f.write('HorizontalContrast,HorizontalFM\n')

		print (pic_name)
		full_img = plt.imread('sceneimages/{}'.format(pic_name))
		m,n,k = full_img.shape
		for a in range(m):
			for b in range(n):
				#Get the necessary images
				color_img, nonblack = circle(a, b, full_img) 
				if nonblack==0: nonblack = 1 #avoid zerodivision errors.
				img = color_img.mean(axis=2) #Convert to B&W 
				vert_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
				horiz_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

				#Get image info
				bright = get_brightness(color_img) / nonblack
				contr = get_RMS_contrast(img) / nonblack
				fm = cv2.Laplacian(img, cv2.CV_64F).var() / nonblack
				vert_contr = get_RMS_contrast(vert_img) / nonblack
				vert_fm = cv2.Laplacian(vert_img, cv2.CV_64F).var() / nonblack
				horiz_contr = get_RMS_contrast(horiz_img) / nonblack
				horiz_fm = cv2.Laplacian(horiz_img, cv2.CV_64F).var() / nonblack

				with open('{}.csv'.format(pic_name.split('.')[0]), 'a') as writefile:
					writefile.write('{},{},{},'.format(pic_name, a, b))
					writefile.write('{},{},{},'.format(bright, contr, fm))
					writefile.write('{},{},'.format(vert_contr, vert_fm))
					writefile.write('{},{},\n'.format(horiz_contr, horiz_fm)) 
					#^can't really do horizontal or vertical brightness, because
					# it relies on the 3 channels, but edge detection with an RGB image is a nightmare.
					# maybe just use mean pixel value for those ones? idk

	return 0
			


if __name__ == '__main__':
	df = pd.read_csv('Scene Viewing Fix Report.txt', delimiter='\t')
	df_1 = df[['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_DURATION', 'picture', 'CURRENT_FIX_INDEX']]
	df_1 = df_1[df_1['picture']!='slum2.jpg']
	
	p = main(df_1)
	
		
		
	
