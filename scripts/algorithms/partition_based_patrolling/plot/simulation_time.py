import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
bots = [i+1 for i in range(15)]
time = [1.570,3.072,4.902,7.116,9.775,12.198,15.136,17.763,21.305,25.232,28.219,32.909,36.303,40.997,45.345]
poly = lagrange(bots,time)
x_new = np.arange(1,15.1,0.1)
y_new = Polynomial(poly.coef[::-1])(x_new)
print(type(y_new),type(x_new))
fig = go.Figure()
fig.add_trace(go.Scatter(x=bots,y=time,mode='lines'))
# fig.add_trace(go.Scatter(x=x_new,y=y_new,mode='lines'))
fig.add_trace(go.Scatter(x=x_new,y=np.interp(x_new,bots,time),mode='lines'))

fig.show()