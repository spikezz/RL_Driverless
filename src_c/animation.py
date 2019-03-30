#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:34:57 2019

@author: spikezz
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from math import sin, cos
from scipy.integrate import odeint

#import matplotlib.animation as animation

# 打开画图窗口1，在三维空间中绘图
#fig = plt.figure()
#
#x = np.linspace(-10, 10, 5)
# 
#y = x 
#
#x0=0
#y0=0
#x1=x
#y1=y
#
## 使用 plot 绘制折线图
## linestyle 设置线的类型，
## 需要注意的 linestyle，c，marker
##for x_ in x:
##    
##    plt.plot([0,1], [0,3], linestyle = "--")
##    
#plt.plot([0,1], [0,3], linestyle = "--")
#plt.plot([0,1], [0,2], linestyle = "--")
#plt.arrow(0,0,0.5,0.3,width=0.01)
#plt.show()
#
#
#
#fig, ax = plt.subplots()
#
##x = np.arange(0, 2*np.pi, 0.01)
#
##line, = ax.plot(x, np.sin(x))
#
#x = np.arange(-10, 10, 0.01)
#line, = ax.plot(x, x)
#
#def animate(i):
##    line.set_ydata(np.sin(x + i/10.0))  # update the data
#    line.set_ydata(x + i/10.0)  # update the data
#    
#    return line,
#
#
## Init only required for blitting to give a clean slate.
#def init():
#    
##    line.set_ydata(np.sin(x))
#    line.set_ydata(x)
#    return line,
#
#
#ani = animation.FuncAnimation(fig=fig, func=animate, frames=2, init_func=init,
#                              interval=20, blit=False)
#
#
#plt.show()




g = 9.8
leng = 1.0
b_const = 0.2

# no decay case:
def pendulum_equations1(w, t, l):
    th, v = w
    dth = v
    dv  = - g/l * sin(th)
    return dth, dv

# the decay exist case:
def pendulum_equations2(w, t, l, b):
    th, v = w
    dth = v
    dv = -b/l * v - g/l * sin(th)
    return dth, dv

t = np.arange(0, 20, 0.1)
#track = odeint(pendulum_equations1, (1.0, 0), t, args=(leng,))
track = odeint(pendulum_equations2, (1.0, 0), t, args=(leng, b_const))
xdata = [leng*sin(track[i, 0]) for i in range(len(track))]
ydata = [-leng*cos(track[i, 0]) for i in range(len(track))]

fig, ax = plt.subplots()
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    time_text.set_text('')
    return line, time_text

def update(i):
    newx = [0, xdata[i]]
    newy = [0, ydata[i]]
    line.set_data(newx, newy)
    time_text.set_text(time_template %(0.1*i))
    return line, time_text

ani = animation.FuncAnimation(fig, update, range(1, len(xdata)), init_func=init, interval=50)
#ani.save('single_pendulum_decay.gif', writer='imagemagick', fps=100)
ani.save('single_pendulum_nodecay.gif', writer='imagemagick', fps=100)
plt.show()

