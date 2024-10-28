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

# x_new_us = 3200
# x_new = (x_new_us - x_min) / (x_max - x_min)  # Scale the new input value
# predicted_price = F(initial_w, initial_b, x_new)



x_new_us = 1750 
x_new = (x_new_us - x_min) / (x_max - x_min)  # Scale the new input value
predicted_price = F(initial_w, initial_b, x_new)
print(f"Predicted price for {x_new_us} sq ft house: {predicted_price} (in thousands of dollars)")

# # Plotting the real training data
# plt.scatter(x_train_us, y_train, color='blue', label='Training Data')  # Original training data (unscaled)
# plt.xlabel('House Area (sq ft)')
# plt.ylabel('Price (in $1000s)')
# plt.title('Real Training Data')
# plt.legend()
# plt.grid(True)
# plt.savefig('real_training_data.png')  # Save the plot as an image
# plt.close()  # Close the plot to free memory

# # Plotting the predicted price for the new area
# plt.scatter([x_new_us], [predicted_price], color='red', label='Predicted Price for 3200 sq ft')  # Prediction point
# plt.xlabel('House Area (sq ft)')
# plt.ylabel('Price (in $1000s)')
# plt.title('Predicted Price for 3200 sq ft House')
# plt.legend()
# plt.grid(True)
# plt.savefig('predicted_price_3200sqft.png')  # Save the plot as an image
# plt.close()  # Close the plot to free memory

# Plotting the real training data and predicted values together
plt.plot(x_train_us, y_train, 'bo-', label='Training Data')  # Systematic x-axis values for training data

# Generate predicted values for all training points
y_pred_all = F(initial_w, initial_b, x_train)

# Plot the predicted line
plt.plot(x_train_us, y_pred_all, 'r-', label='Model Prediction')  # Predicted values as a line

# Mark the specific prediction for 3200 sq ft
plt.scatter([x_new_us], [predicted_price], color='green', label='Predicted Price for 3200 sq ft',zorder=5)  # Prediction point

# Labels and title
plt.xlabel('House Area (sq ft)')
plt.ylabel('Price (in $1000s)')
plt.title('Real vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.savefig('predicted_price_3200sqft.png')  # Save the plot as an image
plt.close()  # Close the plot to free memory