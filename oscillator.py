#coding by Willhelm
#20190309
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# def oscillator(I, u, v, z):
#     a1, a2, a3, a4 = 0.6, 0.6, -0.5, 0.5
#     b1, b2, b3, b4 = -0.6, -0.6, -0.5, 0.5
#     k = 50
#     u_v = np.tanh(a1*u + a2*v - a3*z + a4*I)
#     v_v = np.tanh(b1*z - b2*u - b3*v + b4*I)
#     w = np.tanh(I)
#     z_v = ( v_v - u_v )* np.exp(-k*I*I)+ w
#     return z_v, u_v, v_v


def oscillator(I, u, v, z):
    a1, a2, a3, a4 = 1, 1,1, 1
    b1, b2, b3, b4 = -1, -1, -1, -1
    k = 50
    u_v = np.tanh(a1*u + a2*v - a3*z + a4*I)
    v_v = np.tanh(b1*z - b2*u - b3*v + b4*I)
    w = np.tanh(I)
    z_v = ( v_v - u_v )* np.exp(-k*I*I)+ w
    return z_v, u_v, v_v


# def oscillator(I, u, v, z):
#     a1, a2, a3, a4 = 1, 1,1, 1
#     b1, b2, b3, b4 = -1, -1, -1, -1
#     k = 300
#     u_v = np.tanh(a1*u + a2*v - a3*z + a4*I)
#     v_v = np.tanh(b1*z - b2*u - b3*v + b4*I)
#     w = np.tanh(I)
#     z_v = ( v_v - u_v )* np.exp(-k*I*I)+ w
#     return z_v, u_v, v_v


#working proofed
# def oscillator(I, u, v, z): 
#     a1, a2, a3, a4 = 0.55, 0.55,-0.5, 0.5
#     b1, b2, b3, b4 = -0.55, -0.55, 0.5, -0.5
#     k = 50
#     u_v = np.tanh(a1*u + a2*v - a3*z + a4*I)
#     v_v = np.tanh(b1*z - b2*u - b3*v + b4*I)
#     w = np.tanh(I)
#     z_v = ( v_v - u_v )* np.exp(-k*I*I)+ w
#     return z_v, u_v, v_v

x = np.arange(-150,150)
x = x/150
N = 150
# for every single element
for i in range(0,len(x)):
    #each element in the time sequence T
    print('\r', "progress ",str(i/len(x)).ljust(10),end='')
    u, v, z = np.zeros(N),np.zeros(N),np.zeros(N)
    for time in range(0,N-1):
        z[time+1], u[time+1],v[time+1] = oscillator(x[i], u[time], v[time], z[time])
        plt.plot(x[i], z[time+1], 'c,')
plt.show()



# matrix approach
# x = np.random.rand(50,5)
# u,v,z = 0,0,0
# for i in range(0,len(x)):
#     z,u,v = oscillator(x[i], u,v,z)

