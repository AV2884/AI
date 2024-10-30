import numpy as np
from pprint import pprint

#size_sqft, num_bedrooms, house_age, distance_city
num_features = 4
num_samples = 10
x_train_us = np.array([
    [850, 2, 12, 8.5], 
    [1250, 3, 5, 2.0],
    [1600, 4, 15, 3.2],
    [900, 2, 8, 7.1],
    [2100, 5, 10, 1.5],
    [1450, 3, 4, 6.0],
    [1100, 3, 6, 9.3],
    [950, 2, 7, 12.0],
    [3000, 5, 1, 2.5],
    [2700, 4, 3, 3.8]
])
y_train_us = np.array([30000, 50000, 62000, 31000, 80000, 55000, 42000, 32000, 95000, 88000])
#Normalizing the data
print("Normalizing the data") 
x_normalized = np.zeros_like(x_train_us,dtype=float)
for j in range(num_features):
    mean_x = np.mean(x_train_us[:,j])
    max_x = np.max(x_train_us[:,j])
    min_x = np.min(x_train_us[:,j])
    delta_x = max_x - min_x
    for i in range(num_samples):
        normalized_xi = (x_train_us[i][j] - mean_x)/delta_x
        x_normalized[i,j] = normalized_xi
# print(x_normalized)

y_normalized = np.zeros_like(y_train_us,dtype=float)
mean_y = np.mean(y_train_us[:])
max_y = np.max(y_train_us[:])
min_y = np.min(y_train_us[:])
delta_y = max_y - min_y

for i in range(num_samples):
    normalized_yi = (y_train_us[i] - mean_y)/delta_y
    print(f"{y_train_us[i]} - {mean_y} / {delta_y} = {normalized_yi}")
    y_normalized[i] = normalized_yi
# print(y_normalized)

#model
def F(w, b, x):
    return np.dot(w,x) + b

#cost function
def cost_function(w , b):
    cost = 0
    for i in range(num_samples):
        predicted_value = F(w , b , x_normalized[i])
        error = predicted_value - y_normalized[i]
        cost += pow(error , 2)
    cost = cost / (2 * num_samples)
    return cost

#Gradient descent
def gradient_descent(y, w ,b , alpha , num_iterations):
    M = len(y)

    for epoch in range(num_iterations):
        dw = np.zeros_like(w)
        db = 0  
 
        for i in range(M):
            prediction = F(w=w , b=b , x=x_normalized[i])
            error = prediction - y_normalized[i]

            for j in range(num_features):
                dw[j] += error * x_normalized[i][j] 
            db += error
        
        for j in range(num_features):
            w[j] = w[j] - alpha * dw[j] / M  
        b = b - alpha * db / M

        if epoch % 100 == 0:
            print(f"ITER : {epoch} Cost : {cost_function(w,b)}")

    return w,b

def predict_unnormalized(x, w, b):
    # Predict on normalized input
    normalized_prediction = F(w, b, x)

    # Unnormalize the prediction
    unnormalized_prediction = normalized_prediction * delta_y + mean_y
    return unnormalized_prediction

#________________________________________________________________________________________
w = np.zeros(num_features)  
b = 0  
alpha = 0.0001  
num_iterations = 1_00_00_000  

# Run gradient descent
w, b = gradient_descent(y_normalized, w , b, alpha, num_iterations)

# Print final weights and bias
print("Final weights:", w)
print("Final bias:", b)
#________________________________________________________________________________________


for i in range(num_samples):
    # Get the original (non-normalized) input values for the sample
    area, bedrooms, house_age, distance = x_train_us[i]

    # Predict the price for the normalized input
    prediction = predict_unnormalized(x_normalized[i], w, b)

    # Print the input criteria along with the predicted and actual prices
    print(f"Sample {i+1}:")
    print(f"  Area: {area} sqft, Bedrooms: {bedrooms}, Age: {house_age} years, Distance from city: {distance} km")
    print(f"  Predicted Price: ${prediction:.2f}, Actual Price: ${y_train_us[i]}")
    print("-" * 50)  # Separator for readability
         
#size_sqft, num_bedrooms, house_age, distance_city
# Take user inputs and convert them to float
while(True):
    size_sqft = float(input("Enter size of house: "))
    num_bedrooms = int(input("Enter number of bedrooms: "))
    house_age = float(input("Enter house age: "))
    distance_city = float(input("Enter distance from city: "))

    # Create a NumPy array from the inputs
    usecase = np.array([size_sqft, num_bedrooms, house_age, distance_city], dtype=float)

    # Normalize the input (same normalization logic as your training data)
    usecase_normalized = (usecase - np.mean(x_train_us, axis=0)) / (np.max(x_train_us, axis=0) - np.min(x_train_us, axis=0))

    # Make the prediction using the normalized input
    bill = predict_unnormalized(usecase_normalized, w, b)

    # Print the final prediction
    print(f"Your house will cost approximately ${bill:.2f}")
