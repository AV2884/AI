import numpy as np
import matplotlib.pyplot as plt

x_train_us = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2600, 3000, 3500, 4000, 4500, 5000])  # sq ft
x_min = np.min(x_train_us)
x_max = np.max(x_train_us)
x_train = (x_train_us - x_min) / (x_max - x_min)
y_train = np.array([220, 300, 360, 450, 500, 600, 670, 720, 900, 1050, 1200, 1350, 1500])         # $ in 1000's     # $ in 1000's
m = len(x_train)    

# model
def F(w, b, x):
    return (w * x) + b

# cost function
def J(w,b):
    cost = 0

    for i in range(m):
        cost += pow((F(w , b, x_train[i]) - y_train[i]) , 2)

    cost = (1/(2 * m)) * cost
    return cost


def compute_gradient(w, b):
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        error = (F(w , b, x_train[i]) - y_train[i])
        dj_dw += error * x_train[i]
        dj_db += error

    dj_dw = (1 / m) * dj_dw
    dj_db = (1 / m) * dj_db
    return dj_dw, dj_db

def gradient_descent(w ,b, alpha, num_iters):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(w,b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        if _ % 100 == 0:
            cost = J(w, b)
            print(f"Iteration {_}: Cost {cost}, w: {w}, b: {b}")
    return w , b

# Test compute_gradient with some initial values
initial_w = 1272.7018462263502
initial_b = 223.92670157595194
learning_rate = 0.0001
num_iterations = 1000000000

#For training uncomment:
# # Call gradient_descent function
# final_w, final_b = gradient_descent(initial_w, initial_b, learning_rate, num_iterations)

# print(f"Final values: w = {final_w}, b = {final_b}")
# cost = J(final_w,final_b)
# print(f"final cost :  {cost}")

x_new_us = 3200
x_new = (x_new_us - x_min) / (x_max - x_min)  # Scale the new input value
predicted_price = F(initial_w, initial_b, x_new)
print(f"Predicted price for 3200 sq ft house: {predicted_price} (in thousands of dollars)")

#just implement the plots below here -->