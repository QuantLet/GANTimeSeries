import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from datetime import datetime


class Crypto():
	def __init__(self,path):
		self.data = pd.read_json(path).head(201)
		self.data['ret'] = self.data.price.pct_change().dropna()
		

	def ts_image_and_rp_plot(self,n_rows,n_cols,epsilon):
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.seq_len = n_cols
		self.epsilon = epsilon
		
		# 1 sequentialize the data
		seq_cont = []
		rec_mats = []
		y_images = []
		seqs = []
		for idx in range(self.data.ret.shape[0]-self.seq_len-1):
			print(idx)
			ts = self.data.ret.values[idx:idx+self.seq_len]
			seq_cont.append(ts)
			
			if (idx+1>=self.seq_len):
				print(str(idx)+'/'+str(self.data.ret.shape[0]-self.seq_len-1))
				ts_matrix = np.array(seq_cont)[-self.seq_len:]
				
				# create a recurrence plot
				R = np.zeros((self.seq_len,self.seq_len))

				for x in range(self.seq_len):
					for y in range(self.seq_len):
						R[x,y] = 1 if np.abs(ts[x]-ts[y])<epsilon else 0

				# reset the seqs_container

				y_images.append(self.data.ret.values[idx+self.seq_len])
				seqs.append(ts_matrix)
				rec_mats.append(R)

				

				self.seqs = seqs
				self.y_images = y_images
				self.rec_mats = rec_mats

				font = {'family': 'sans-serif',
								'color':  'red',
								'weight': 'bold',
								'size': 13,
								}

				fig, ax = plt.subplots(figsize=(12,8))	
				
				#Plot1
				sns.heatmap(R,ax=ax,cmap='Blues',vmin=0,vmax=1)
				
				ax.set_xticks(np.arange(0,n_rows,5).tolist())
				ax.set_yticks(np.arange(0,n_rows,5).tolist())
				ax.set_title('Recurrence Plot with '+chr(949)+'='+str(self.epsilon)[:5],fontdict=font)
				ax.invert_yaxis()
				plt.savefig('e'+str(self.epsilon)+'_rp.png')


				fig, ax3 = plt.subplots(figsize=(12,8))	
				ax3.plot(self.data['date'][np.arange(idx,idx+self.seq_len)],ts)
				# Hide the right and top spines
				ax3.spines['right'].set_visible(False)
				ax3.spines['top'].set_visible(False)

				# Only show ticks on the left and bottom spines
				ax3.yaxis.set_ticks_position('left')
				ax3.xaxis.set_ticks_position('bottom')

				ax3.set_title('CRIX',fontdict=font)
				plt.tight_layout()
				plt.savefig('ts.png',transparent=True)
				plt.close()
				

crix = Crypto('crix.json')
es = np.arange(0,0.036,0.001)


for eps in es:
	crix.ts_image_and_rp_plot(100,100,eps)
	