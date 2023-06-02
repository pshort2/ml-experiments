import numpy as np
import copy, math
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Activation functions (g)

#Sigmoid function
def sigmoid(z):

    g = 1/(1+np.exp(-z))
   
    return g

#relu function
def relu(z):

	zeros = np.zeros(len(z))
	g = np.maximum(zeros,z)

	return g

def softmax(z):  
           
    a = np.exp(z)/np.sum(np.exp(z))
    
    return a

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Cost function for logistic regression with regularisation
def cost_log(X, y, w, b, g, lambda_ = 0):

	m = X.shape[0]
	z = np.dot(X,w) + b
	
	f_wb = g(z)
    
	reg = (lambda_/(2*m)) * np.dot(w,w) #regularisation term 
    
	cost =  ((1/m) * np.sum( -y*np.log(f_wb) - (1-y)*np.log(1-f_wb) )) + reg
             
	cost /= m
    
	return cost


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Calculate gradient with regularisation
def gradient(X, y, w, b, g, lambda_=0): 

	m = X.shape[0]
	dj_dw = np.zeros((n,))                           

	z = np.dot(X,w) + b
	f_wb = g(z)
    
	reg = (lambda_/m) * np.sum(w) #regularisation term
                    
	dj_dw = (1/m) * np.dot(f_wb - y, X) + reg
	dj_db = (1/m) * np.sum(f_wb - y)
                               
        
	return dj_db, dj_dw  


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Perform gradient descent with logistic regression
def gradient_descent(X, y, w_in, b_in, alpha, n, g, lambda_=0): 

	J_history = []
	w = copy.deepcopy(w_in)  #avoid modifying global w within function
	b = b_in
    
	for i in range(n):
		# Calculate the gradient and update the parameters
		dj_db, dj_dw = gradient(X, y, w, b, g, lambda_)   

		# Update Parameters using w, b, alpha and gradient
		w = w - alpha * dj_dw               
		b = b - alpha * dj_db               
      
		# Save cost J at each iteration
		if i<100000:      # prevent resource exhaustion 
			J_history.append(cost_log(X, y, w, b, g, lambda_) )

		# Print cost every at intervals 10 times or as many iterations if < 10
		if i% math.ceil(n / 10) == 0:
			print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
		return w, b, J_history


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Computes layer of NN, equivalent to Dense function in Tensorflow
def my_dense(a_in, W, b, g):
                                            
	z = np.dot(np.transpose(W), a_in) + b
	
	a_out = g(z)
         
	return(a_out)
