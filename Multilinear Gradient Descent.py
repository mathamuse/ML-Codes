############################## INCLUSIONS AND HEADERS #########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################# Create Euclidean cost function ##################
def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))

############################ MultiLinear Gradient Descent Function ############ 
def gradient_descent_multi(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)
    gdm_df = pd.DataFrame( columns = ['Bets','cost'])

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        gdm_df.loc[i] = [theta,cost]

    return gdm_df

# Theta is the vector representing coefficients (intercept, area, bedrooms)
theta = np.matrix(np.array([0,0,0])) 
alpha = 0.0001
iterations = 100

############################## Read file and plot #############################

df = pd.read_csv("C:/... filepath ...../Data for Multilinear Regression 3.csv")

print(df)                                    # mainly for testing 

#'''
X = df.iloc[:,:-1]                           # split data
y= df.iloc[:,-1]                             # split outcome

#print(" X is : ",X )                        # mainly for testing
#print( " y  is :",y )                       # mainly for testing

######## call the gradient descent function ###############
gdf =gradient_descent_multi(X, y, theta, alpha, iterations)
#print(gdf)
#'''
###############################################################################
##                                 OUTPUT                                    ##
###############################################################################
cost = gdf['cost']
#print(cost)
for l in range(iterations):
    values = np.array(gdf.iloc[l,0])
    y_pred = []
    x1_pred=[]
    for i in range(len(X)):
    #print(X.iloc[i,:])
        y_pred.append(np.dot(values,X.iloc[i,:]))
        #x1_pred.append(np.dot(values,X.iloc[i,:])
         
    #print(y_pred)
    #''' #Uncomment this section to see a dynamic plot of the grad desc   
    plt.plot(y_pred,color='red')
    plt.plot(y)
    plt.show()
    #'''


# Calculate the differences (residuals)
errors = y - pd.Series(y_pred)

# Square the errors
squared_errors = errors ** 2

# Calculate the Mean Squared Error (MSE)
mse = squared_errors.mean()

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("\n RMSE is : ",rmse)
################# Plot Cost Score as Quality control ###################

plt.plot(cost)
plt.xlabel("Iterations")
plt.ylabel("Cost function")
#'''
