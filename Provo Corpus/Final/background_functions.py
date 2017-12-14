from __future__ import print_function
import pandas as pd 
from matplotlib import pyplot as plt 
import matplotlib
import sys
import numpy as np 
from pandas.tools.plotting import scatter_matrix
plt.style.use('ggplot')

def get_df():
	df = pd.read_csv('fixation_report.csv')
	# yesfix = df[df['Fixation'] > 0]
	# nofix = df[df['Fixation'] == 0]
	# print (len(yesfix), len(nofix))
	return df

def scat(df):
	scatter_matrix(df)
	plt.savefig('scatter_matrix.png')

def color_plot(df):
	# colors = ['Red', 'Green', 'Blue']
	cols = [i for i in df.columns if i not in ['picture', 'short_picture']]
	colors = get_xkcd_colors()

	for ind, c in enumerate(cols):
		plt.subplot(4,4,ind+1)
		try:
			if c=='Y': raise(ValueError)
			df[c].plot(kind='hist', by=c, color=c.lower(), bins=20, ylim=[0,13000], sharey=True)
		except:
			df[c].plot(kind='hist', by=c, color='gray', bins=20, ylim=[0,13000], sharey=True) #if the column name isn't a name of a color.
		
		plt.title(c)
	# plt.suptitle('Distribution of Features in Images', fontsize=16, weight='bold')
	plt.tight_layout()

def bright_plot(df):
	df['Brightness'].plot(kind='hist', by='Brightness', color='gray', bins=20)
	plt.title('Brightness')

def distr_plot(df):
	# plt.subplot(311)
	df['Contrast'].plot(kind='hist', by='Contrast', color='y', bins=50)
	plt.title('Contrast')
	plt.show()
	# plt.subplot(312)
	df['HorizContrast'].plot(kind='hist', by='HorizContrast', color='c', bins=50)
	plt.title('Horizontal Contrast')
	plt.show()
	# plt.subplot(313)
	df['VertContrast'].plot(kind='hist', by='VertContrast', color='m', bins=50)
	plt.title('Vertical Contrast')	
	plt.show()

def fixation_and_color(df):
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,10))
	colors = ['Red', 'Green', 'Blue']
	avg_fix = df['Fixation'].mean()
	for ind, c in enumerate(colors):
		cuts = []
		for i in np.linspace(0,1,11):
			cuts.append(df[c].quantile(q=i))

		k = pd.cut(df[c], bins=cuts)
		l = pd.pivot_table(df, values='Fixation', index=k)
		l.plot(kind='barh', color=c, ax=axes[ind])
		axes[ind].axvline(x=avg_fix, c='gray', label='Average: {}'.format(round(avg_fix, 1)))
		axes[ind].legend()
		axes[ind].yaxis.label.set_visible(False)
	plt.suptitle('How does color affect fixation?', weight='bold', fontsize=24)

def get_xkcd_colors():
	cool_colors = pd.read_csv('~/Desktop/xkcd_colors.txt', delimiter='\t', skiprows=1, header=None)
	cool_colors.columns = ['name', 'code', 'nan']
	return cool_colors['code'].values

def scatter_fix(df, deg=1):
	avg_fix = df['Fixation'].mean()
	l = ['Contrast', 'HorizContrast', 'VertContrast']
	fixes = df['Fixation'].values

	plt.subplot(221)
	contr = df['Contrast'].values
	plt.scatter(contr, fixes, color='gray', alpha=0.6)
	z1 = np.polyfit(contr, fixes, deg)
	f1 = np.poly1d(z1)
	x1 = np.linspace(contr.min(), contr.max(), 100)
	plt.plot(x1, f1(x1), 'r')
	plt.title('Contrast')

	plt.subplot(222)
	hocontr = df['HorizContrast'].values
	plt.scatter(hocontr, fixes, color='gray', alpha=0.6)
	z2 = np.polyfit(hocontr, fixes, deg)
	f2 = np.poly1d(z2)
	x2 = np.linspace(hocontr.min(), hocontr.max(), 100)
	plt.plot(x2, f2(x2), 'g')
	plt.title('Horizontal Contrast')

	plt.subplot(223)
	vertcontr = df['VertContrast'].values
	plt.scatter(vertcontr, fixes, color='gray', alpha=0.6)
	z3 = np.polyfit(vertcontr, fixes, deg)
	f3 = np.poly1d(z3)
	x3 = np.linspace(vertcontr.min(), vertcontr.max(), 100)
	plt.plot(x3, f3(x3), 'b')
	plt.title('Vertical Contrast')

	plt.subplot(224)
	big = np.max([contr.max(), hocontr.max(), vertcontr.max()])
	small = np.min([contr.min(), hocontr.min(), vertcontr.min()])
	print (contr.max(), hocontr.max(), vertcontr.max())
	print (big, small)
	x = np.linspace(small, big, 100)
	# plt.plot(x, f1(x), 'r', label='Contrast')
	plt.plot(x, f2(x), 'g', label='Horizontal Contrast')
	plt.plot(x, f3(x), 'b', label='Vertical Contrast')


if __name__ == '__main__':
	df = get_df()
	print (df.columns)
	print (len(df.columns))