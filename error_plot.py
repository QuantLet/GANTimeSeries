import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap

y = np.load('y_test.npy')[:-1]
yhat = np.load('preds_test.npy')[1:]
dates = np.load('dates.npy')[-y.shape[0]:]



break_1 = 125
break_2 = 300
break_3 = 350
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dates[:break_1], y[:break_1],c='blue')
ax.plot(dates[break_1-1:break_2], y[break_1-1:break_2],c='green')
ax.plot(dates[break_2-1:break_3], y[break_2-1:break_3], c='blue')
ax.plot(dates[break_3-1:], y[break_3-1:], c='green')
ax.plot(dates, yhat, c='red')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('split_sequence.png', transparent=True)

ccoeff = np.corrcoef(yhat.flatten(), y.flatten())[1, 0].round(4)

print(ccoeff)
N = 200
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1,0.1646, N)
vals[:, 1] = np.linspace(1,0.2658, N)
vals[:, 2] = np.linspace(1,0.5696, N)
newcmp = ListedColormap(vals)

cm = plt.cm.get_cmap('Blues')

yhat_break = yhat[:break_1]
y_break = y[:break_1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(yhat_break, y_break, c=np.arange(y_break.shape[0]), vmin=0, vmax=y_break.shape[0],cmap=cm)
ax.plot(np.linspace(min(y), max(y), yhat_break.shape[0]), np.linspace(min(y), max(y), yhat_break.shape[0]), color='Gray')
ax.set_xlabel(r'$\hat{y}(t)$')
ax.set_ylabel(r'$y(t-1)$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('error_plot_CNN_up1.png',transparent=True)
plt.show()

cm = plt.cm.get_cmap('Greens')
yhat_break = yhat[break_1:break_2]
y_break = y[break_1:break_2]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(yhat_break, y_break, c=np.arange(y_break.shape[0]), vmin=0, vmax=y_break.shape[0],cmap=cm)
ax.plot(np.linspace(min(y), max(y), yhat_break.shape[0]), np.linspace(min(y), max(y), yhat_break.shape[0]), color='Gray')
ax.set_xlabel(r'$\hat{y}(t)$')
ax.set_ylabel(r'$y(t-1)$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('error_plot_CNN_down1.png', transparent=True)
plt.show()

cm = plt.cm.get_cmap('Blues')
yhat_break = yhat[break_2-1:break_3]
y_break = y[break_2-1:break_3]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(yhat_break, y_break, c=np.arange(y_break.shape[0]), vmin=0, vmax=y_break.shape[0],cmap=cm)
ax.plot(np.linspace(min(y), max(y), yhat_break.shape[0]), np.linspace(min(y), max(y), yhat_break.shape[0]), color='Gray')
ax.set_xlabel(r'$\hat{y}(t)$')
ax.set_ylabel(r'$y(t-1)$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('error_plot_CNN_up2.png', transparent=True)
plt.show()

cm = plt.cm.get_cmap('Greens')
yhat_break = yhat[break_3-1:]
y_break = y[break_3-1:]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(yhat_break, y_break, c=np.arange(y_break.shape[0]), vmin=0, vmax=y_break.shape[0],cmap=cm)
ax.plot(np.linspace(min(y), max(y), yhat_break.shape[0]), np.linspace(min(y), max(y), yhat_break.shape[0]), color='Gray')
ax.set_xlabel(r'$\hat{y}(t)$')
ax.set_ylabel(r'$y(t-1)$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('error_plot_CNN_down2.png', transparent=True)
plt.show()