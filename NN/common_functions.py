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

	w1, b1, w2, b2, w3, b3 = params[0], params[1], params[2], params[3], params[4], params[5]

    z1 = np.dot(w1, x_train) + b1
    a1 = relu(z1)
    
	z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)

	z3 = np.dot(w3, a2) + b3

    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Carries out backward propagation
def backward_prop(x_train, y_train, activations, params):

    m = X.shape[1]
    a1, a2, a3 = activations["a1"], activations["a2"]
    w3 = params[4]

    dz3 = a3 - y_train #error in final layer
    dw3 = np.dot(dz3, a2.T) / m #derivative of w3
    db3 = np.sum(dz3, axis=1, keepdims=True) / m #derivative of b3, 'keepdims' keeps the dimensions of db3 the same as dz2

    dz2 = np.dot(w3.T, dz3) * relu_derivative(a2)
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

	dz1 = np.dot(w2.T, dz2) * relu_derivative(a2)
    dw2 = np.dot(dz1, x_train.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2, "dw3": dw3, "db3": db3}
