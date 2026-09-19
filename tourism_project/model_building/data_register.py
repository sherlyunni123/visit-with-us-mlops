import os
import pandas as pd
DATA_PATH = "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS=['CustomerID',
                  'ProdTaken',
                  'Age',
                  'TypeofContact',
                  'CityTier', 
                  'DurationOfPitch',
                  'Occupation', 
                  'Gender', 
                  'NumberOfPersonVisiting', 
                  'NumberOfFollowups', 
                  'ProductPitched',
                  'PreferredPropertyStar',
                  'MaritalStatus', 
                  'NumberOfTrips', 
                  'Passport', 
                  'PitchSatisfactionScore', 
                  'OwnCar', 
                  'NumberOfChildrenVisiting',
                  'Designation', 
                  'MonthlyIncome'
                ]

def register_dataset():
  if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
  df=pd.read_csv(DATA_PATH)

  missing_columns=[
      column for column in EXPECTED_COLUMNS
      if column not in df.columns
  ]
  if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )
  print(f"Dataset registered successfully")
  print(f"Dataset shape: {df.shape}")
  print(f"Number of columns: {len(df.columns)}")
  print(f"Target column: ProdTaken")
  print("\nTarget distribution:")
  print(df["ProdTaken"].value_counts())
if __name__ == "__main__":
  register_dataset()
