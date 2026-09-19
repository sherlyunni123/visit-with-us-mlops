import os
import pandas as pd
from sklearn.model_selection import train_test_split
#configuration

DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR="tourism_project/model_building"

TRAIN_FILE=os.path.join(OUTPUT_DIR,"Xtrain.csv")
TEST_FILE=os.path.join(OUTPUT_DIR,"Xtest.csv")
YTRAIN_FILE=os.path.join(OUTPUT_DIR,"ytrain.csv")
YTEST_FILE=os.path.join(OUTPUT_DIR,"ytest.csv")


#Load dataset
print(f"Loading dataset from :{DATA_PATH}")
df=pd.read_csv(DATA_PATH)
print(f"Dataset shape :{df.shape}")
df=df.drop(columns=['Unnamed: 0','CustomerID'])

#remove ProdTaken
X=df.drop(columns=['ProdTaken'])
y=df['ProdTaken']
X.shape,y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#recreate the train and test files

# save the feature files locally
X_train.to_csv(TRAIN_FILE, index=False)
X_test.to_csv(TEST_FILE, index=False)

# save target files
y_train.to_csv(YTRAIN_FILE, index=False, header=True)
y_test.to_csv(YTEST_FILE, index=False, header=True)

#Display the results
print("\nData loaded completed successfully")
print(f"Training data {TRAIN_FILE}")
print(f"Testing data {TEST_FILE}")
print(f"Train shape :{X_train.shape}")
print(f"Test shape :{X_test.shape}")
print(f"Training target: {YTRAIN_FILE}")
print(f"Testing target: {YTEST_FILE}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
