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

	zeros = np.zeros(z.shape)
	g = np.maximum(zeros,z)

	return g

#relu derivative for back prop
def relu_derivative(z):

    dz[z <= 0] = 0
    dz[z > 0] = 1

    return dz

#softmax function
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
                                        
	z = np.dot(a_in,W) + b
	
	a_out = g(z)
         
	return(a_out)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Carries out forward propagation
def forward_prop(x_train, params):

	'''
	arguments:
	x_train - training data
	params - dictionary of parameters W1 ... Wn and b1 ... bn (weights and biases)

	returns:
	AL - output of final layer
	outputs - list of outputs for each layer
	'''

	outputs = [] 
	A = x_train 

	L = len(params) // 2 #this is the number of layers of the NN (divided by 2 as the dictionary contains both weights and biases
	
	for l in range(1, L):

		A_prev = A #Store the previous activation
		W = params["W" + str(l)] #for the first layers calls W1, the second layer W2 etc...
		b = params["b" + str(l)]

		Z = np.dot(W, A_prev) + b
		A = relu(Z) #use relu for all layers other than final
    
		output = (A_prev, W, b, Z)
		outputs.append(output)

	#For the final layer:
	AL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)] #linear activation for final layer
	output = (A, parameters['W' + str(L)], parameters['b' + str(L)], AL)
	outputs.append(output)

	return AL, outputs

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Carries out backward propagation
def backward_prop(AL, y_train, outputs):

	'''
	arguments:
	AL - output of forward prop
	y_train - training y data

	returns:
	grads - dictionary of gradients
	'''

	grads = {}
	m = AL.shape[1] #this is the number of training samples
	L = len(outputs) #this is the number of layers

	#Initialisation of back prop
	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	#Back prop for the final layer (linear activation)
	dZL = dAL
	A_prev, W, b, Z = outputs[L - 1]

	grads['dW' + str(L)] = np.dot(dZL, A_prev.T) / m
	grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m

	dA_prev = np.dot(W.T, dZL)
	
	#Back prop for the oher layers
	for l in reversed(range(L - 1)):

		A_prev, W, b, Z = outputs[l]
		dZ = relu_backward(dA_prev, Z)

		grads['dW' + str(l + 1)] = np.dot(dZ, A_prev.T) / m
		grads['db' + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True) / m

		dA_prev = np.dot(W.T, dZ)

	return grads
