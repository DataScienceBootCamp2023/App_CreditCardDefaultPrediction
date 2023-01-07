import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

class CreditCardDefaultPrediction():
    def __init__(self, dtf_input):
      self.dtf_input = dtf_input

    @staticmethod
    def predict(self):
    
           # Load the model from a file
        with open('saved_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Use the model to Make predictions on the input data
        predictions = model.predict(self.dtf_input.iloc[:, 1:-1])
        # Create a dataframe with the predictions and the original data
        dtf_predictions = pd.concat([self.dtf_input.iloc[:, 0], pd.DataFrame(predictions, columns=["default"])], axis=1)
        dtf_predictions["default"] = dtf_predictions["default"].apply(lambda x: "Yes" if x==1 else "No"

        return dtf_predictions

    @staticmethod
    def write_excel(dtf_predictions):
        bytes_file = io.BytesIO()
        excel_writer = pd.ExcelWriter(bytes_file)
        dtf_predictions.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', index=False)
        excel_writer.save()
        bytes_file.seek(0)
        return bytes_file


