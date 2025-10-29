# catboost_model.py

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming DataManager and calculate_scaled_MAE are in utils.py
from utils import DataManager, calculate_scaled_MAE


class CatBoost:
    """
    A wrapper class for training and evaluating a CatBoost model.
    
    This class follows the same interface as the LightBGM class:
    1. Inference Mode (default): Initialize with `CatBoost()`.
    2. Training Mode: Initialize with `CatBoost.from_training_data(...)`.
    """

    def __init__(self, random_state: int = 42):
        """
        Initializes the CatBoost trainer in a lightweight,
        inference-ready state.
        
        To initialize with data for training, use the
        `CatBoost.from_training_data()` classmethod.

        Args:
            random_state (int): The seed for all random operations.
        """
        self.data_manager = None
        self.random_state = random_state
        self.cat_features = []
        
        self.model = None
        
        self.base_params = {
            'learning_rate': 0.03,
            'depth': 12,
            'l2_leaf_reg': 1.0,
            'subsample': 0.8, 
            'colsample_bylevel': 0.8,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': random_state,
            'thread_count': -1,
        }
        print("CatBoost trainer initialized in base (inference) mode.")

    @classmethod
    def from_training_data(cls,
                           path: str,
                           test_size: float = 0.1,
                           val_size: float = 0.15,
                           random_state: int = 42):
        """
        Class method to create and initialize a CatBoost instance
        with data for a full training pipeline.

        Args:
            path (str): The file path to the dataset.
            test_size (float): Proportion of data for the test split.
            val_size (float): Proportion of train data for validation.
            random_state (int): The seed for all random operations.
        
        Returns:
            CatBoost: A class instance loaded with data, ready for training.
        """
        instance = cls(random_state=random_state)
        
        print("Initializing DataManager for CatBoost...")
        instance.data_manager = DataManager(
            path=path,
            model_type="catboost",  # Critical: tells DataManager to prep for CatBoost
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        
        # Store the categorical feature names provided by DataManager
        instance.cat_features = instance.data_manager.cat_features
        print(f"DataManager loaded. Found {len(instance.cat_features)} categorical features.")
        return instance

    def train_final_model(self, save_path = "final_catboost_model.cbm"):
        """
        Trains the final CatBoost model.
        
        This process involves two steps:
        1. Train on (Train) / (Val) with early stopping to find the
           best number of iterations.
        2. Train a new model on the full (Train + Val) dataset using
           that specific number of iterations.

        Args:
            save_path: Path to save the final model.
        """
        if self.data_manager is None:
            raise ValueError("Cannot train. DataManager not initialized. Use `from_training_data()` to initialize.")

        print("\n--- STAGE 1: Finding best iteration ---")
        print("NOT FINALIZED - SHOULD BE DONE WITH GRID SEARCH - FOR NOW: best-guess values")
        
        # Create data pools (required for CatBoost)
        train_pool = Pool(
            self.data_manager.X_train,
            self.data_manager.y_train,
            cat_features=self.cat_features
        )
        val_pool = Pool(
            self.data_manager.X_val,
            self.data_manager.y_val,
            cat_features=self.cat_features
        )

        temp_model = CatBoostRegressor(
            **self.base_params,
            iterations=4000, 
            logging_level='Verbose',
            cat_features=self.cat_features # Pass this for correct handling
        )
        
        temp_model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=200
        )
        
        best_iteration = temp_model.get_best_iteration()
        if best_iteration is None:
            print("Warning: Early stopping did not trigger. Using max iterations (4000).")
            best_iteration = 4000
            
        print(f"Found best iteration: {best_iteration}")

        print("\n--- STAGE 2: Training final model on (Train + Val) data ---")
        
        train_val_pool = Pool(
            self.data_manager.X_train_val,
            self.data_manager.y_train_val,
            cat_features=self.cat_features
        )
        
        self.model = CatBoostRegressor(
            **self.base_params,
            iterations=best_iteration, # Use the best iteration
            logging_level='Verbose',
            cat_features=self.cat_features
        )
        
        self.model.fit(train_val_pool, verbose=200)
        print("Successfully fit the final model!")

        # Save the model
        if save_path:
            try:
                print(f"Saving final model to {save_path}...")
                self.model.save_model(save_path, format="cbm")
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving model: {e}")

    def evaluate_model(self):
        """Evaluates the final model on the Train-Val and Test datasets."""
        if self.data_manager is None:
            raise ValueError("Cannot evaluate. DataManager not initialized.")
        if self.model is None:
            raise ValueError("Model has not been trained yet. Run `train_final_model()` first.")
        
        # Create pools for evaluation
        train_val_pool = Pool(self.data_manager.X_train_val, cat_features=self.cat_features)
        test_pool = Pool(self.data_manager.X_test, cat_features=self.cat_features)

        # Evaluate on the (Train + Val) set
        train_val_log_preds = self.model.predict(train_val_pool)
        calculate_scaled_MAE(
            self.data_manager.y_train_val,
            train_val_log_preds,
            test=False
        )
        
        # Evaluate on the Test set
        final_log_preds = self.model.predict(test_pool)
        calculate_scaled_MAE(
            self.data_manager.y_test,
            final_log_preds,
            test=True
        )

    def plot_feature_importance(self, top_n: int = 15):
        """Plots the top N feature importances from the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Run `train_final_model()` first.")
        
        importances = self.model.get_feature_importance()
        feature_names = self.model.get_feature_names()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, max(6, top_n // 2.5)))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=importance_df.head(top_n)
        )
        plt.title(f'Top {top_n} Feature Importances (from CatBoost)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def run_catboost_pipeline(self, top_n_features: int = 15,
                              model_save_path: str = "final_catboost_model.cbm"):
        """
        Runs the complete end-to-end training and evaluation pipeline.(No tuning step for this version).
        Args:
            top_n_features (int): The number of top features to plot.
            model_save_path (Optional[str]): Path to save the final model.
        """
        if self.data_manager is None:
            raise ValueError("Cannot run pipeline. DataManager not initialized. Use `from_training_data()` to initialize.")
            
        print("======== STARTING CATBOOST PIPELINE ========")
        print("\n--- STAGE 1: FINAL MODEL TRAINING ---")
        self.train_final_model(save_path=model_save_path)
        print("\n--- STAGE 2: MODEL EVALUATION ---")
        self.evaluate_model()
        print("\n--- STAGE 3: FEATURE IMPORTANCE ---")
        self.plot_feature_importance(top_n=top_n_features)
        print("\n======== CATBOOST PIPELINE COMPLETE ========")

    def predict(self, data_path: str, pretrained_model_path: str = None) -> float:
        """
        Loads data, preprocesses it for CatBoost, and returns a prediction.

        Args:
            data_path (str): File path to the CSV with the data point.
            pretrained_model_path (Optional[str]): Path to a saved CatBoost
                                                   model file (.cbm).
        Returns:
            float: The final, un-logged (scaled to dollars) prediction.
        """
        model_to_use = None
        if pretrained_model_path:
            print(f"Loading pretrained model from {pretrained_model_path}...")
            try:
                model_to_use = CatBoostRegressor()
                model_to_use.load_model(pretrained_model_path)
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                raise
        elif self.model:
            print("Using model trained in this session...")
            model_to_use = self.model
        else:
            raise ValueError(
                "No model available. Either run a training pipeline first "
                "or provide a 'pretrained_model_path'."
            )

        print(f"Loading data from {data_path} for prediction...")
        try:
            data_point_df = pd.read_csv(data_path, low_memory=False)
            if len(data_point_df) > 1:
                print(f"Warning: data_path contains {len(data_point_df)} rows. "
                      "Predicting first row only.")
                data_point_df = data_point_df.head(1)
            
            processed_df = data_point_df.copy()
            
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    processed_df[col] = processed_df[col].fillna("Missing").astype(str) 
        
            all_features = model_to_use.get_feature_names()
            predict_pool = Pool(processed_df, cat_features=all_features)
            
            # Prediction
            log_pred = model_to_use.predict(predict_pool)
            final_pred = np.expm1(log_pred)
            
            return float(final_pred[0])

        except FileNotFoundError:
            print(f"Error: File not found at {data_path}")
            raise
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise