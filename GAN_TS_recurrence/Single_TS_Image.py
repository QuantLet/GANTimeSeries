import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from datetime import datetime


class Crypto():
	def __init__(self,path):
		self.data = pd.read_json(path).head(9)
		self.data.date = self.data['date'].dt.strftime('%m-%d-%Y')

		

	def scale_data(self,a,b):
		self.a = a
		self.b = b
		self.minimum = np.min(self.data.price)
		self.maximum = np.max(self.data.price)

		self.data['price_sc'] = a+((self.data.price - self.minimum)*(b-a))/(self.maximum-self.minimum) 
		print('min value in df:'+str(np.min(self.data.price_sc)))
		print('max value in df:'+str(np.max(self.data.price_sc)))
		print(self.data['price_sc'])

	def ts_image(self,n_rows,n_cols):
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.seq_len = n_cols
		
		# 1 sequentialize the data
		seq_cont = []
		rec_mats = []
		y_images = []
		seqs = []
		for idx in range(self.data.price_sc.shape[0]-self.seq_len):
			row = self.data.price_sc.values[idx:idx+self.seq_len]
			
			print('############')
			seq_cont.append(row)
			
			if (idx+1>=self.seq_len):
				print(str(idx)+'/'+str(self.data.price_sc.shape[0]-self.seq_len-1))
				ts_matrix = np.array(seq_cont)[-self.seq_len:]#last 3 rows


				#flip to right dimension
				#ts_matrix = np.flip(ts_matrix,axis=0)
				
				ts=np.concatenate([ts_matrix[0,:],ts_matrix[1:,-1]],axis=0)
				#ts_matrix = np.flip(ts_matrix,axis=0)
				seqs.append(ts_matrix)		

				self.seqs = seqs
				self.y_images = y_images
				self.rec_mats = rec_mats

				font = {'family': 'sans-serif',
								'color':  'red',
								'weight': 'bold',
								'size': 13,
								}



				gridsize = (1,2)
				fig = plt.figure(figsize=(14,5))
				ax1 = plt.subplot2grid(gridsize, (0, 0))
				ax3 = plt.subplot2grid(gridsize, (0, 1))
				
				#Plot1
				sns.heatmap(ts_matrix,ax=ax1,cmap='Blues')
				ax1.set_title('TS Image',fontdict=font)
				ax1.set_xlabel('i')
				ax1.set_ylabel('j')
				ax1.invert_yaxis()
				
				print(self.data['date'])
				print(ts)

				ax3.plot(self.data['date'].head(self.n_cols*2-1),ts.flatten())
				
				ax3.set_xticks(self.data['date'].values[np.arange(0,self.n_cols*2-1,2)].tolist())
				# Hide the right and top spines
				ax3.spines['right'].set_visible(False)
				ax3.spines['top'].set_visible(False)

				# Only show ticks on the left and bottom spines
				ax3.yaxis.set_ticks_position('left')
				ax3.xaxis.set_ticks_position('bottom')

				ax3.set_title('CRIX',fontdict=font)
				plt.savefig('singletsimage.png',transparent=True)
				break
				


crix = Crypto('crix.json')
crix.scale_data(-1,1)
crix.ts_image(4,4)
	