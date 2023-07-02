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
#Cost function for logistic regression (no regularisation yet)
def cost_log(AL, y):

	m = y.shape[0]
    
	cost =  ((1/m) * np.sum( -y[0]*np.log(AL) - (1-y[0])*np.log(1-AL) ))
             
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
def forward_prop(X, params):

	'''
	arguments:
	x_train - training data
	params - dictionary of parameters W1 ... Wn and b1 ... bn (weights and biases)

	returns:
	AL - output of final layer
	outputs - list of outputs for each layer
	'''

	outputs = [] 
	A = X 

	L = len(params) // 2 #this is the number of layers of the NN (divided by 2 as the dictionary contains both weights and biases
	
	for l in range(1, L):

		A_prev = A #Store the previous activation
		W = params["W" + str(l)] #for the first layers calls W1, the second layer W2 etc...
		b = params["b" + str(l)]

		Z = np.dot(A_prev, W) + b
		A = relu(Z) #use relu for all layers other than final
    
		output = (A_prev, W, b, Z)
		outputs.append(output)

	#For the final layer:
	AL = np.dot(A, params['W' + str(L)]) + params['b' + str(L)] #linear activation for final layer
	output = (A, params['W' + str(L)], params['b' + str(L)], AL)
	outputs.append(output)

	return AL, outputs

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Carries out backward propagation
def backward_prop(AL, y, outputs):

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
	dAL = -(np.divide(y[0], AL) - np.divide(1 - y[0], 1 - AL))

	#Back prop for the final layer (linear activation)
	dZL = dAL
	A_prev, W, b, Z = outputs[L - 1]

	grads['dW' + str(L)] = (np.dot(dZL.T, A_prev) / m).T
	grads['db' + str(L)] = np.sum(dZL, axis=0, keepdims=True) / m

	dA_prev = np.dot(dZL, W.T)
	
	#Back prop for the oher layers
	for l in reversed(range(L - 1)):

		A_prev, W, b, Z = outputs[l]
		dZ = copy.deepcopy(dA_prev) #avoid modifying global dA within function
		dZ [Z <= 0]	#set gradients to 0 where Z <= 0 for the relu derivative

		grads['dW' + str(l + 1)] = (np.dot(dZ.T, A_prev) / m).T
		grads['db' + str(l + 1)] = np.sum(dZ, axis=0, keepdims=True) / m

		dA_prev = np.dot(dZ, W.T)

	return grads

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Update parameters using the learning rate and gradients
def update_params(params, grads, alpha):
	'''
	arguments:
	params - dictionary of parameters W1 ... Wn and b1 ... bn (weights and biases)
	grads - gradients
	alpha - learning rate

	returns:
	The updated parameters
	'''
	L = len(params) // 2

	for l in range(1, L):

		params["W" + str(l)] -= alpha * grads["dW" + str(l)] #for the first layers calls W1, the second layer W2 etc...
		params["b" + str(l)] -= alpha * grads["db" + str(l)][0]

	return params

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Train the neural network
def train_nn(X, y, params, alpha, n):

	for i in range(n):

		print ("Iteration: " + str(i) + "/" + str(n))
		
		AL, outputs = forward_prop(X, params)
		cost = cost_log(AL, y)
		print ("cost:", cost)

		grads = backward_prop(AL, y, outputs)
		params = update_params(params, grads, alpha)

	return params
		
