import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap


def recurrence_plot(ts,window_size,epsilon):
	R = np.zeros((window_size,window_size))

	for x in range(window_size):
		for y in range(window_size):
			R[x,y] = 1 if np.abs(ts[x]-ts[y])<epsilon else 0
	return(R)

def AR1_process(alpha,x0,mu,sigma_e,n_steps):

	x = [x0]
	for step in range(n_steps):
		x.append(alpha*x[step-1]+np.random.normal(mu,sigma_e)) #alt+5 for brackets
	x = np.array(x)
	return(x)

def sin(start,end,num_steps):
	x = np.linspace(start, end, 100)
	return(x,np.sin(x))

def wn(n_steps):
	seq = [0]
	for idx in range(n_steps):
		seq.append(np.random.normal(0,1))
	return(seq)

def make_transparent(file):
	img = Image.open(file)
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
	    if item[0] == 255 and item[1] == 255 and item[2] == 255:
	        newData.append((255, 255, 255, 0))
	    else:
	        newData.append(item)

	img.putdata(newData)
	img.save(file, "PNG")
	
font = {'family': 'sans-serif',
				'color':  'red',
				'weight': 'bold',
				'size': 13,
				}

np.random.seed(5)
#Create custom color map

N = 200
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1,0.1646, N)
vals[:, 1] = np.linspace(1,0.2658, N)
vals[:, 2] = np.linspace(1,0.5696, N)
newcmp = ListedColormap(vals)


################ AR1  ####################
AR1 = AR1_process(0.8,0,0,0.1,1000)
R_AR1 = recurrence_plot(AR1,100,0.1)

gridsize = (2,1)
fig = plt.figure(figsize=(5,10))
ax1 = plt.subplot2grid(gridsize, (1, 0))
ax2 = plt.subplot2grid(gridsize, (0, 0))

#Plot1
sns.heatmap(R_AR1,ax=ax1,cmap=newcmp)
ax1.set_title('RP',fontdict=font)
ax1.set_xlabel('i')
ax1.set_ylabel('j')
ax1.invert_yaxis()

ax2.plot(AR1[:100],color=(0.1646,0.2658,0.5696))

# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

ax2.set_title('AR1',fontdict=font)

plt.savefig('AR1.png',transparent=True)
plt.close()
make_transparent('AR1.png')
############### Sinus ####################

x,sin = sin(0,20,100)

R_sin = recurrence_plot(sin,100,0.1)

gridsize = (2,1)
fig = plt.figure(figsize=(5,10))
ax1 = plt.subplot2grid(gridsize, (1, 0))
ax2 = plt.subplot2grid(gridsize, (0, 0))

#Plot1
sns.heatmap(R_sin,ax=ax1,cmap=newcmp)
ax1.set_title('RP',fontdict=font)
ax1.set_xlabel('i')
ax1.set_ylabel('j')
ax1.invert_yaxis()

ax2.plot(sin,color=(0.1646,0.2658,0.5696))

# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

ax2.set_title('sin',fontdict=font)
plt.savefig('sinus.png',transparent=True)
plt.close()
make_transparent('sinus.png')
########### White Noise ###################

wn = wn(100)

R_wn= recurrence_plot(wn,100,0.1)

gridsize = (2,1)
fig = plt.figure(figsize=(5,10))
ax1 = plt.subplot2grid(gridsize, (1, 0))
ax2 = plt.subplot2grid(gridsize, (0, 0))

#Plot1
sns.heatmap(R_wn,ax=ax1,cmap=newcmp)
ax1.set_title('RP',fontdict=font)
ax1.set_xlabel('i')
ax1.set_ylabel('j')
ax1.invert_yaxis()

ax2.plot(wn,color=(0.1646,0.2658,0.5696))

# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

ax2.set_title('White Noise',fontdict=font)
plt.savefig('wn.png',transparent=True)
plt.close()
make_transparent('wn.png')






