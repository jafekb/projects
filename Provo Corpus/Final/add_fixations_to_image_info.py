import pandas as pd 
import numpy as np 
import sys
import os
from tqdm import tqdm

#Don't try this at home, kids
import warnings
warnings.filterwarnings("ignore")

def round_down(num, divisor):
    return num - (num%divisor)

def get_image_info():
	image_info = pd.read_csv('FULL_image_info.csv')
	#Get just the info of the image which relates to the full picture, not the partition.
	image_info['short_picture'] = [i.split('_')[0].lower() for i in image_info['picture']] 

	return image_info

def get_fixation_info():
	fixations = pd.read_csv('Scene Viewing Fix Report.txt', delimiter='\t')
	fix = fixations[['picture', 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_DURATION']]
	fix.dropna(subset=['picture'], inplace=True) #There are a few images which don't have labeled images; we can't work with that.
	fix['short_picture'] = [i.split('.')[0].lower() for i in fix['picture']] 
	fix['short_picture'] = [i.split('_')[0] for i in fix['short_picture']] 
	fix['CURRENT_FIX_X'] -= 400
	fix['CURRENT_FIX_Y'] -= 150
	fix = fix[(fix['CURRENT_FIX_X'] >= 0) & (fix['CURRENT_FIX_X'] <= 600)] #get rid of extraneous X data
	fix = fix[(fix['CURRENT_FIX_Y'] >= 0) & (fix['CURRENT_FIX_Y'] <= 800)] #get rid of extraneous Y data

	return fix

def include():
	fixes = get_fixation_info()
	img_info = get_image_info()
	img_info['Fixation'] = 0

	for i in tqdm(fixes.values):
		pic, x_loc, y_loc, duration, short_name = i
		x_bottom = round_down(x_loc, 25)
		y_bottom = round_down(y_loc, 25)
		img_info['Fixation'][(img_info['X']==x_bottom) & (img_info['Y']==y_bottom)] += 1 #don't worry about duration just yet.
		#TODO maybe on later iterations take both (1) duration, and (2) index into account

	img_info.to_csv('FULL_fixation_report.csv', index=False)
	return img_info

if __name__ == '__main__':
	a = include()
	print (a)