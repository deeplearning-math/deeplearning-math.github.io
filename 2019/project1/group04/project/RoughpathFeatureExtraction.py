import pandas as pd
import iisignature as iis
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from datetime import datetime

# Read the data, save the number of ticks
Data=pd.read_csv(r'C:\Data\DLProject\BTCUSDT_hf.csv')
t=500


n=Data.shape[0]-5000


#  We restrict our model to the most important predictors. We do this for two reasons:
#  1) Computational cost of calculating signatures of a 45 dimensional path is to high for a personal computer
#  2) We would obtain  a very large number of predictors. Therefore the model would be prone to overfitting,
#     since the number of predictors is large and the signal to noise ratio is low.

Data=Data[['time', 'price', 'qty', 'side']]


# We process the timestamps to seconds in order to be able to calculate the singatures of the paths
time=pd.to_datetime(Data['time'],format='%Y-%m-%d %H:%M:%S,%f')
seconds=np.zeros(Data.shape[0])


for i in range (Data.shape[0]):
    seconds[i]=(time[i]-time[0]).total_seconds()
    print(i)
Data['time']=seconds



# We fill the NaN with a forward filling method.
# After that we fill the remaining NaN with backward filling method. Note that there are only very few NaNs
# left after doing the forward filling. We do this to get full data. Note that we only backwardfill values which are before the first prediction.
Data=Data.fillna(method='ffill')
Data=Data.fillna(method='bfill')

# The variable 'side' is a categorical variable. We use get.dummies to turn it into integer values
Data['side']=pd.get_dummies(Data['side'],drop_first=True)

#We're now ready to start computing signatures. The package iisignature has automated the functions we need

sig=[]
y=[]

for i in range(int(n)):
    newsig=iis.sig(Data[1000+i:1000+i+t],3)
    newy=Data['price'][1000+i+t+60]-Data['price'][1000+i+t]
    y=np.append(y,newy)
    sig=np.concatenate((sig,newsig))
    print(i,'/',int(n))

#We reshape the result and save it in a Dataframe
sig=np.reshape(sig,(int(n),-1))
pandasig=pd.DataFrame(sig)
ysig=pd.DataFrame(y)

# We save the obtained features and target variables in a CSV

pandasig.to_csv(r'C:\Data\sigwtime.csv', header=None, index=False)
ysig.to_csv(r'C:\Data\y.csv', header=None, index=False)











