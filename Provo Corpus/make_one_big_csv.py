import glob
import pandas as pd 
import sys
from tqdm import tqdm
import os
import numpy as np 


def make_combined():
	csvs = glob.glob('CSVs/*.csv')
	csvs.extend(glob.glob('CSV2/*.csv'))
	csvs.remove('CSVs/computerlab4.csv')
	csvs.remove('CSV2/lockerroom4.csv')
	csvs.remove('CSV2/marsh1.csv')
	csvs.remove('CSV2/moun_tains.csv')

	if os.path.isfile('combined.csv'):
		os.system('rm combined.csv')

	df = pd.read_csv(csvs[0])
	cols = df.columns
	arr = df.values


	for ind in tqdm(range(1, len(csvs))):
		data = pd.read_csv(csvs[ind]).values
		print (data.shape, arr.shape, csvs[ind])
		arr = np.append(arr, data, axis=0)

	dfout = pd.DataFrame(arr, columns=cols)

	dfout.to_csv('combined.csv', index=False)

def see_combined():
	df = pd.read_csv('combined.csv')
	print (df.head())
	return

def add_fixations():
	fixations = pd.read_csv('fixations.txt', delimiter='\t')
	fixations = fixations[fixations['picture']=='FOUNTAIN.jpg']
	all_points = pd.read_csv('combined.csv')
	all_points = all_points[all_points['Picture']=='FOUNTAIN.jpg']
	all_points.to_csv('just_fountain.csv', index=False)
	print (len(fixations), len(all_points))
	all_points['Duration'] = 0
	all_points['Looked'] = 0
	c=0
	for i in fixations.values:
		dur, ind, x, y, pic = i
		x -= 400
		y -= 150
		all_points['Duration'][(all_points['X_center']==x) & (all_points['Y_center']==y) & (all_points['Picture']==pic)] = dur
		all_points['Looked'][(all_points['X_center']==x) & (all_points['Y_center']==y) & (all_points['Picture']==pic)] = 1

		if all_points[(all_points['X_center']==x) & (all_points['Y_center']==y) & (all_points['Picture']==pic)].empty:
			print (dur, ind, x, y, pic)
			print ('x:', all_points[(all_points['X_center']==x)].empty)
			print ('y:', all_points[(all_points['Y_center']==y)].empty)
			print ('pic:', all_points[(all_points['Picture']==pic)].empty)
			print;
		c+=1
		# if c>100:break

	check = all_points[all_points['Duration']!=0]
	print (len(check))
	# all_points.to_csv('final_fixations_and_combined.csv', index=False)

if __name__ == '__main__':
	print(add_fixations())