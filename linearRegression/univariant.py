import numpy as np
import time
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
initial_w = 0
initial_b = 0
learning_rate = 0.0001
num_iterations = 1_00_00_000

#For training uncomment---------------------------------------------------------------
start_time = time.time()
initial_cost = J(initial_w, initial_b)
final_w, final_b = gradient_descent(initial_w, initial_b, learning_rate, num_iterations)
end_time = time.time()
final_cost = J(final_w, final_b)
time_taken = end_time - start_time
x_new_us = 1000
x_new = (x_new_us - x_min) / (x_max - x_min)
predicted_price = F(final_w, final_b, x_new)
cost = J(final_w,final_b)
print("\n=== Summary ===")
print(f"Training Time: {time_taken:.4f} seconds")
print(f"Initial Cost: {initial_cost:.4f}")
print(f"Final Cost: {final_cost:.4f}")
print(f"Cost Reduced By: {((initial_cost - final_cost)/initial_cost)*100:.2f}%")
print(f"Final Weights: w = {final_w:.4f}, b = {final_b:.4f}")
print(f"Prediction for {x_new_us} sq ft: {predicted_price:.2f} (in thousands of dollars)")
#--------------------------------------------------------------------------------------
# x_new_us = 1000 
# x_new = (x_new_us - x_min) / (x_max - x_min)  # Scale the new input value
# w = 1272.4309201554588
# b = 224.04991993731397
# predicted_price = F(w, b, x_new)

