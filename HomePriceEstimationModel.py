import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import datetime

#Synthetic training data because I didn't look for a real dataset. I created a csv file with these random numbers that I'll include
'''
np.random.seed(0)
x = pd.DataFrame({
    'SquareFootage': np.random.randint(1000, 4501, 100),
    'NumberOfBedrooms': np.random.randint(2, 8, 100),
    'NumberOfBathrooms': np.random.randint(1, 6, 100),
    'YearBuilt': np.random.randint(1980, 2025, 100),
    'Stories': np.random.randint(1, 4, 100),
    'HasGarage': np.random.randint(0, 2, 100)  #Dummy variable for garage
})
'''

x = pd.read_csv('HomeData.csv')

#Create a new column, HouseAge, needed due to issues with YearBuilt messing up home value estimations
current_year = datetime.datetime.now().year
x['HouseAge'] = current_year - x['YearBuilt']
x = x.drop(columns=['YearBuilt'])

#Base home value for home with 0 sq ft, 0 bedrooms, etc. Adds/reduces value based on criteria. Adds random noise to account for real world fluctuations 
y = 50000 + 200 * x['SquareFootage'] + 10000 * x['NumberOfBedrooms'] + 5000 * x['NumberOfBathrooms'] - 10000 * x['HouseAge'] + 10000 * x['Stories'] + 5000 * x['HasGarage'] + np.random.normal(0, 10000, 100)

#If the training data had prices associated with each house, use this to define y instead:
    #y = x['Price']
    #x = x.drop(columns=['Price'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['SquareFootage', 'NumberOfBedrooms', 'NumberOfBathrooms', 'HouseAge', 'Stories', 'HasGarage'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(x_train, y_train)

'''
#Debugging section: Needed this to figure out that YearBuilt was an issue and switched to HouseAge instead. Commented out cause I don't need this printing anymore but I still want it documented
print("\nModel Coefficients:")
print(model.named_steps['regressor'].coef_)
print("\nModel Intercept:")
print(model.named_steps['regressor'].intercept_)
'''

#Regression Table and other diagnostic value(s)
y_train_pred = model.predict(x_train)
print('\nTraining Mean Squared Error:', mean_squared_error(y_train, y_train_pred))
x_train_sm = sm.add_constant(x_train)  #Adds a constant term for the intercept
model_sm = sm.OLS(y_train, x_train_sm).fit()
print()
print(model_sm.summary())

#Is using all these while True loops the best way to do this?
def evaluate_house_price(model):
    print("\nEnter the house details -")
    while True:
        try:
            square_footage = float(input("Square Footage: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid amount for square footage.")

    while True:
        try:
            num_bedrooms = int(input("Number of Bedrooms: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid amount for the number of bedrooms.")

    while True:
        try:
            num_bathrooms = int(input("Number of Bathrooms: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid amount for the number of bathrooms.")

    while True:
        try:
            year_built = int(input("Year Built: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid year for the year built.")

    while True:
        try:
            stories = int(input("Number of Floors: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid amount for the number of floors.")

    while True:
        garage_input = input("Does the house have a garage? (y/n): ").strip().lower()
        if garage_input in ['y', 'yes']:
            has_garage = 1
            break
        elif garage_input not in ['y', 'yes']:
            has_garage = 0
            break
    
    #Calculate HouseAge, needed due to issues with YearBuilt messing up home value estimations
    current_year = datetime.datetime.now().year
    house_age = current_year - year_built
    
    # Create a DataFrame for the input data
    user_data = pd.DataFrame({
        'SquareFootage': [square_footage],
        'NumberOfBedrooms': [num_bedrooms],
        'NumberOfBathrooms': [num_bathrooms],
        'HouseAge': [house_age],
        'Stories': [stories],
        'HasGarage': [has_garage]
    })
    
    ''' #This was also part of figuring out the issue with YearBuilt
    #Manually scale the data using the pipeline's scaler for comparison
    scaled_user_data = model.named_steps['preprocessor'].transform(user_data)
    print("\nScaled user input data:")
    print(scaled_user_data)
    '''
    #Predict house price
    predicted_price = model.predict(user_data)[0]
    print(f"\nThe predicted fair listing price for this house is: ${predicted_price:,.2f}")
    
    user_input2 = input("\nDo you want to check if a specific price is fair? (y/n): ").strip().lower()
    if user_input2 in ['yes', 'y']:
        while True:
            try:
                input_price = float(input("Enter the price you want to check this house at: ").replace(",", ""))
                break
            except ValueError:
                print('Invalid input. Please enter a valid number.')

        #Evaluate the fairness of the input price
        if input_price < predicted_price * 0.9:
            print(f"The price ${input_price:,.2f} is below market value.")
        elif input_price > predicted_price * 1.1:
            print(f"The price ${input_price:,.2f} is above market value.")
        else:
            print(f"The price ${input_price:,.2f} is fair based on the market.")


user_input = input("\nDo you want to evaluate a specific house price? (y/n): ").strip().lower()
if user_input in ['yes', 'y']:
    evaluate_house_price(model)
else:
    pass



#Some graphs to help check model effectiveness, I have them commented out at the moment since they are cluttering up my console lol
'''
# Plot Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# Distribution of Predicted Prices
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Predicted Prices')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Prices')
plt.show()
'''
