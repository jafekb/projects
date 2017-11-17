import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt 
import cv2

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

def circle(center_x, center_y, img, func):
	print (center_x, center_y)
	r = 75 #This relates pretty well to how much a human sees.	

	l = img.shape
	m, n = l[0], l[1] #It also might have a depth dimension
	arr = np.zeros_like(img)

	for i in range(m): #I couldn't figure out how to mask this as a circle...
		for j in range(n):
			if (center_x - j)**2 + (center_y - i)**2 <= r**2:
				arr[i,j] = img[i,j]

	return func(img)

def main(df):
	for point in df.values:
		x, y, dur, pic_name, index = point
		x -= 400
		y -= 150

		img = plt.imread('sceneimages/{}'.format(pic_name))
		bright = get_brightness(img)

		img = img.mean(axis=2) #Convert to B&W
		contr = get_RMS_contrast(img)
		
		vert_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=0)
		vert_contr = get_RMS_contrast(vert_img)

		horiz_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=0)
		horiz_contr = get_RMS_contrast(horiz_img)		
		
		sys.exit(0)
	
	return 0


if __name__ == '__main__':
	df = pd.read_csv('Scene Viewing Fix Report.txt', delimiter='\t')
	df_1 = df[['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_DURATION', 'picture', 'CURRENT_FIX_INDEX']]
	
	main(df_1)
	
		
		
	
