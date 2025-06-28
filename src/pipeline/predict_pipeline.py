import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_obj

class CustomData:
    def __init__(self, amount: float,
                 mode: str,
                 subcategory: str):
        self.amount = amount
        self.mode= mode
        self.subcategory = subcategory

    def get_df(self):
        data = {
            'Amount': [self.amount],
            'Mode': [self.mode],
            'Subcategory': [self.subcategory]
        }

        return pd.DataFrame(data)
    
class PredictPipeline:
    def __init__(self):
        ...

    def predict(self, input: pd.DataFrame):
        preprocessor = load_obj('artifacts/preprocessor.pkl')
        model = load_obj('artifacts/model.pkl')
        label_encoder = load_obj('artifacts/label_encoder.pkl')

        print("Input columns received by preprocessor:", input.columns.tolist())

        data_scaled = preprocessor.transform(input)
        preds = model.predict(data_scaled)
        print(preds)
        decoded_preds = label_encoder.inverse_transform(preds.astype(int))

        return decoded_preds