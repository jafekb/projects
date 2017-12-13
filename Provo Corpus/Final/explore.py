from __future__ import print_function
import pandas as pd 
from matplotlib import pyplot as plt 
import matplotlib
import sys
import numpy as np 
from pandas.tools.plotting import scatter_matrix
plt.switch_backend('QT4Agg')

def get_df():
	df = pd.read_csv('FULL_fixation_report.csv')
	# yesfix = df[df['Fixation'] > 0]
	# nofix = df[df['Fixation'] == 0]
	# print (len(yesfix), len(nofix))
	return df

def scat(df):
	scatter_matrix(df)
	manager = plt.get_current_fig_manager()
	manager.window.showMaximized()
	plt.savefig('scatter_matrix.png')

def color_plot(df):
	colors = ['Red', 'Green', 'Blue']
	for ind, c in enumerate(colors):
		plt.subplot(1,3,ind+1)
		df[c].plot(kind='hist', by=c, color=c.lower(), bins=20, sharey=True, ylim=[0,13000])
		plt.title(c)
	plt.suptitle('Distribution of Color in Images', fontsize=16, weight='bold')
	plt.show()

def fixation_and_color(df):
	colors = ['Red', 'Green', 'Blue']

	avg_fix = df['Fixation'].mean()
	for ind, c in enumerate(colors):
		cuts = []
		for i in np.linspace(0,1,11):
			cuts.append(df[c].quantile(q=i))

		k = pd.cut(df[c], bins=cuts)
		l = pd.pivot_table(df, values='Fixation', index=k)
		l.plot(kind='barh', color=c)
		plt.axvline(x=avg_fix, c='gray', label='Average: {}'.format(round(avg_fix, 1)))
		plt.title('How does {} affect Fixations?'.format(c.lower()), weight='bold')
		plt.legend()
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.savefig('Graphs/{}_pivot_bar.png'.format(c.lower()))
		plt.clf()

def contrast_fixation(df):
	contrasts = ['Contrast', 'HorizContrast', 'VertContrast']
	colors = 'cmy'
	avg_fix = df['Fixation'].mean()

	for ind, c in enumerate(contrasts):
		print (ind, c)
		cuts = []
		for i in np.linspace(0,1,11):
			cuts.append(df[c].quantile(q=i))

		k = pd.cut(df[c], bins=cuts)
		l = pd.pivot_table(df, values='Fixation', index=k)
		l.plot(kind='barh', color=colors[ind])
		plt.axvline(x=avg_fix, c='gray', label='Average: {}'.format(round(avg_fix, 1)))
		plt.title('How does {} affect Fixations?'.format(c.lower()), weight='bold')
		plt.legend(loc='lower right')
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.savefig('Graphs/{}_pivot_bar.png'.format(c.lower()))
		plt.clf()

def generic_fixation_pivot_grapher(df, factor_list, colors='cmy'):
	avg_fix = df['Fixation'].mean()

	for ind, c in enumerate(factor_list):
		print (ind, c)
		cuts = []
		for i in np.linspace(0,1,11):
			cuts.append(df[c].quantile(q=i))

		k = pd.cut(df[c], bins=cuts)
		l = pd.pivot_table(df, values='Fixation', index=k)
		l.plot(kind='barh', color=colors[ind])
		plt.axvline(x=avg_fix, c='gray', label='Average: {}'.format(round(avg_fix, 1)))
		plt.title('How does {} affect Fixations?'.format(c.lower()), weight='bold')
		plt.legend(loc='lower right')
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.savefig('Graphs/{}_pivot_bar.png'.format(c.lower()))
		plt.clf()

def get_xkcd_colors():
	cool_colors = pd.read_csv('~/Desktop/xkcd_colors.txt', delimiter='\t', skiprows=1, header=None)
	cool_colors.columns = ['name', 'code', 'nan']
	return cool_colors['code'].values

if __name__ == '__main__':
	df = get_df()
	# scat(df)
	# color_plot(df)
	fixation_and_color(df)
	l1 = ['Brightness', 'FM', 'HorizFM', 'VertFM', 'X', 'Y']
	colors = get_xkcd_colors()
	colors = np.random.choice(colors, len(l1))
	generic_fixation_pivot_grapher(df,l1,colors)


