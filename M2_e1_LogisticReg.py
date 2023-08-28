# Fabián González Vera A01367585

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ETL
print('good')
columns = ["class","cap-shape","cap-surface","cap-color","bruises",
           "odor","gill-attachment","gill-spacing","gill-size",
           "gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
           "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring",
           "veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]
df = pd.read_csv('mushroom/agaricus-lepiota.data',names = columns)

temp = df[["cap-shape","cap-surface","cap-color",
           "odor","stalk-shape","stalk-root","stalk-surface-above-ring",
           "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring",
           "spore-print-color","population","habitat"]]

temp1Hot = pd.get_dummies(temp, dtype=float)

df = df[["class"]].join(temp1Hot)

df['class'] = df['class'].apply(lambda letter: 1.0 if letter=='e' else 0.0)

newdf = df.copy()
#print(list(newdf))
newdf = newdf.sample(frac=1)
#slice_df = (1-len(newdf))
#newdf_x = newdf[(1-len(newdf.columns)):]
newdf_x = newdf.loc[:, 'cap-shape_b':'habitat_w']
newdf_y = newdf[['class']]

dflen = len(newdf)
trainS = math.floor(80*dflen/100)
testS = math.floor(20*dflen/100)
train_x = newdf_x[:][:trainS]
train_y = newdf_y[:][:trainS]

test_x = newdf_x[:][-testS:]
test_y = newdf_y[:][-testS:]

print('size: ',testS)
print('dflen: ',dflen)

print(test_y)
print(test_x)
print('lens')
print(len(newdf_y))
print(len(newdf_x))
# Logistic Regression

def h(params, data): #evaluates h(x) = 1/(1+e^-x), f(x) = a+bx1+cx2+ ... nxn..
    #print(params)
    #print(data)
    r = map(lambda x, y: x * y, params, data)
    acum = sum(list(r))
    hx = 1/(1+math.exp(-acum))
    return hx

def GD(params, data, y, alpha):
    print('enter gd, give it time')
    nparams = list(params)
    for j in range(len(params)):
        acum =0
        for i in range(len(data)):
            c_df_row = data.iloc[i]
            error = h(params, c_df_row) - y.iloc[i]
            acum = acum + error * c_df_row[j]  #Sumatory part of the Gradient Descent formula for linear Regression.
            nparams[j] = params[j] - alpha*(1/len(data))*acum  #Subtraction of original value with learning rate included.
    return nparams

def show_errors(params, data, y): #The cost function, in logistic regression is Logistic Cost Function
	global __errors__
	print('enter show errors')
	#print(data)
	#print(y)
	
	#print(params)
	error_acum =0
	error = 0
	"""
	Where "-LOG(0) -> ∞"   and  "-LOG(1-1) -> ∞"
	if   y =  1   
		- LOG ( h ( x ) )
	else  if   y  = 0  
		- LOG ( 1 - h ( x ) )
	"""
	for i in range(len(data)):
		#print('enters for i ', i)
		hyp = h(params, data.iloc[i])
		#print('exits hyp fun')
		#print(type(y.iloc[0]))
		#print(y.iloc[0])
		
		val_y = y.iloc[i].item()
		#print(val_y)
		if(val_y == 1.0): # avoid the log(0) error
			if(hyp ==0.0):
				hyp = .0001;
			error = (-1)*math.log(hyp); # -LOG ( h(x) )
		if(val_y == 0.0):
			if(hyp ==1.0):
				hyp = .9999;
			error = (-1)*math.log(1-hyp); # -LOG(1 - h(x) )
		print( "error %f  hyp  %f  y %f " % (error, hyp,  val_y)) 
		error_acum = error_acum + error # this error is different from the one used to update, this is general for each sentence it is not for each individual param
	#print("acum error %f " % (error_acum));
	mean_error_param=error_acum/len(data);
	#print("mean error %f " % (mean_error_param));
	__errors__.append(mean_error_param)
	return mean_error_param;


# LG setup
__errors__= []

params = np.zeros(len(train_x.columns))#not len of og df because we are adding bias

alfa =.03  #  learning rate
print('b b',len(train_x))
bias = np.ones(len(train_x))
train_x["Bias"] = bias #  Include bias into train & test dataset 

bias = np.ones(len(test_x))
test_x["Bias"] = bias

print(len(params))
print(len(train_x))
print("original samples:")
print(train_x)

epoch = 0
while True: # run gradient descent until local minima is reached
	oldparams = list(params)
	
	print(params)
	params = GD(params, train_x, train_y, alfa)	
	error = show_errors(params, train_x, train_y) # only used to show errors, it is not used in calculation
	print(params)
	params_alt = lambda x: params[x].item(), params
	# print(type(oldparams[0]))
	# print(type(error))
	epoch += 1
	print(epoch)
	# local minima is found when there is no further improvement, stop when error is 0, or epoch = n
	if(oldparams == params_alt or error < 0.1 or epoch == 1): 
		print("Data:")
		print(train_x)
		print("final params:")
		print(params)
		break

# train_x = newdf_x[:][:trainS]
# train_y = newdf_y[:][:trainS]

# test_x = newdf_x[:][-testS:]
# test_y = newdf_y[:][-testS:]

# PREDICTIONS
m_guesses = []
for i in range(len(test_x)):
	c_df_row = test_x.iloc[i]
	y_pred = h(params, c_df_row)
	print("Expected=%.3f, Predicted=%.3f [%d]" % (test_y.iloc[i], y_pred, round(y_pred)))
	m_guesses.append(round(y_pred))

act_y = test_y.to_numpy()
act_y = act_y.flatten()
print(len(test_y))
print(len(test_x))
df_confusion = pd.crosstab(act_y, m_guesses, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)

"""
			pred_NO		pred_YES		total
act_NO		trueNO		falseYES		totalNumberOfNo's
act_YES		falseNO		trueYes			totalNumberOfYes's
total		totNoPreds	totYesPreds		totalNumberOfPreds
"""

plt.plot(__errors__)
plt.show()