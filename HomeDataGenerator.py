import numpy as np
import pandas as pd

# Synthetic data
np.random.seed(0)
df = pd.DataFrame({
    'SquareFootage': np.random.randint(1000, 4501, 100),
    'NumberOfBedrooms': np.random.randint(2, 8, 100),
    'NumberOfBathrooms': np.random.randint(1, 6, 100),
    'YearBuilt': np.random.randint(1980, 2025, 100),
    'Stories': np.random.randint(1, 4, 100),
    'HasGarage': np.random.randint(0, 2, 100)  #Dummy variable for garage
})

print(df)
print("Creating csv...")
df.to_csv('HomePrices.csv', index = False)
print("Done")
