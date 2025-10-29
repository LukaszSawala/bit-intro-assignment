import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class DataManager:
    """
    A class to manage data loading, preprocessing, and splitting for the models.

    This class handles reading data from a CSV file, performing model-specific
    preprocessing for categorical features (LightGBM or CatBoost), and splitting
    the data into training, validation, and test sets.
    """
    def __init__(self, path: str = "data/processed/processed_v1.csv", model_type: str = "lgbm",
                 test_size: float = 0.1, val_size: float = 0.15, random_state: int = 42):
        """
        Initializes the DataManager object.

        Args:
            path (str): The file path to the dataset. Defaults to "data/processed/processed_v1.csv".
            model_type (str): The type of model for which the data is being prepared ('lgbm' or 'catboost'). Defaults to "lgbm".
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.1.
            val_size (float): The proportion of the training set to use for validation. Defaults to 0.15.
            random_state (int): The seed used by the random number generator. Defaults to 42.
        """
        self.load_data(path, model_type)
        self.split_data(test_size, val_size, random_state)
        print("Data successfully loaded and split.")
        

    def load_data(self, path: str = None, model_type: str = None):
        """
        Loads and preprocesses the data from a given path.

        This method reads a CSV file, handles potential file not found errors,
        and preprocesses categorical columns based on the specified model type.
        For 'catboost', it fills NaNs and converts to string. For 'lgbm', it
        converts them to the 'category' dtype. It also separates features (X)
        from the target variable (y). If the dataset is unprocessed, it 
        shifts targets into the logspace.

        Args:
            path (str): The file path to the dataset.
            model_type (str): The type of model for which the data is being prepared ('lgbm' or 'catboost').
        """
        try:
            df = pd.read_csv(path, low_memory=False)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return

        if model_type == "catboost":
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("Missing").astype(str)   # Fill NaNs with the string "Missing" (required for CatBoost)
        elif model_type == "lgbm":
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
        else:
            print("Selected model is wrong. Options: lgbm/catboost")

        self.X = df.drop('Sales Price', axis=1)
        self.y = df['Sales Price']
        if "processed" not in path:     # that means our labels are still exponentially distributed
            self.y = np.log1p(self.y)


        if model_type == "catboost":
            cat_features = self.X.select_dtypes(include=['object']).columns.tolist()
            print(f"Identified {len(cat_features)} categorical features for CatBoost.")
        else:
            cat_features = [] # not useful for lgbm
    
        self.cat_features = cat_features

    def split_data(self, test_size, val_size, random_state):
        """
        Splits the data into training, validation, and test sets.

        The data is first split into a training/validation set and a test set.
        Then, the training/validation set is further split into a final training set and a validation set.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the train_val set to use for validation.
            random_state (int): The seed used by the random number generator for reproducibility.
        """
        assert test_size < 1, "test_size must be between 0 and 1"
        assert val_size < 1, "val_size must be between 0 and 1"
        
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val, test_size=val_size, random_state=42 
        )

        print(f"Total shape:    {self.X.shape}")
        print(f"Train shape:    {self.X_train.shape}")
        print(f"Validate shape: {self.X_val.shape}")
        print(f"Test shape:     {self.X_test.shape}")


def calculate_scaled_MAE(y_true, y_pred, test = True):
    """
    Calculate the MAE by first moving it out of the logspace and then presenting
    """
    y_true_scaled = np.expm1(y_true)
    final_preds = np.expm1(y_pred)

    final_dollar_mae = mean_absolute_error(y_true_scaled, final_preds)

    if test:
        print(f"Final Test MAE (Unbiased): ${final_dollar_mae:,.2f} in % of the price: {100 * final_dollar_mae / y_true_scaled.mean():.2f}%")
    else:
        print(f"MAE for the Train-Val set: ${final_dollar_mae:,.2f} in % of the price: {100 * final_dollar_mae / y_true_scaled.mean():.2f}%")



    
    