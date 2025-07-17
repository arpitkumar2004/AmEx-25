import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

class DataLoader:
    def read_data(folder_path=r"C:\Users\RDRL\Desktop\AmEx 25\data"):
        files = [f for f in os.listdir(folder_path) if f.endswith('.parquet') or f.endswith('.csv')]
        dataframes = {}
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            dataframes[file] = df
        return dataframes

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df, target_column):
        df = df.copy()
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def transform(self, df):
        df = df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        df = df.fillna(df.median(numeric_only=True))
        X_scaled = self.scaler.transform(df)
        return X_scaled

class ModelTrainer:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

class SubmissionGenerator:
    def __init__(self,test_data_path=r"C:\Users\RDRL\Desktop\AmEx 25\data\test_data.parquet"):
        self.test_df = pd.read_parquet(test_data_path)

    def generate(self, output_path=r"C:\Users\RDRL\Desktop\AmEx 25\submission\submission.csv"):
        submission = pd.DataFrame({
            "id1": self.test_df['id1'],
            "id2": self.test_df['id2'],
            "id3": self.test_df['id3'],
            "id5": self.test_df['id5'],
            "pred": self.test_df['pred']
        })
        submission.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")

class Pipeline:

    def __init__(self, train_path, test_path, target_column, submission_path):
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.submission_path = submission_path

    def run(self):
        # Step 1: Load data
        loader = DataLoader(folder_path=r"C:\Users\RDRL\Desktop\AmEx 25\data")
        event_df, trans_df, offer_df, test_df ,train_df, = loader.load_data()

        # Step 2: Preprocess data
        # preprocessor = DataPreprocessor()
        # X, y = preprocessor.fit_transform(train_df, self.target_column)
        # test_ids = test_df['id'] if 'id' in test_df.columns else np.arange(len(test_df))
        # X_test = preprocessor.transform(test_df.drop(columns=['id'], errors='ignore'))

        # # Step 3: Split data for validation
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # # Step 4: Train model
        # trainer = ModelTrainer()
        # trainer.train(X_train, y_train)

        # # Step 5: Evaluate model
        # val_score = trainer.evaluate(X_val, y_val)
        # print(f"Validation Accuracy: {val_score:.4f}")

        # # Step 6: Predict on test data
        # test_preds = trainer.predict(X_test)

        # Step 7: Generate submission
        submission_gen = SubmissionGenerator()
        submission_gen.generate()

if __name__ == "__main__":

    pipeline = Pipeline()
    pipeline.run()