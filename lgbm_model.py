import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from typing import Dict, Any, Optional
from utils import DataManager, calculate_scaled_MAE

# Suppress Optuna's trial logging
#optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBM:
    """
    A wrapper class for training, tuning, and evaluating a LightGBM model.
    
    This class can be used in two modes:
    1. Inference Mode (default): Initialize with `LightGBM()`.
       This is a lightweight object.
    2. Training Mode: Initialize with `LightGBM.from_training_data(...)`.
       This loads the data and prepares the class for a full pipeline run.
    """
    def __init__(self, random_state: int = 42):
        """
        Initializes the LightGBM trainer in a lightweight,
        inference-ready state.
        
        To initialize with data for training, use the
        `LightGBM.from_training_data()` classmethod.

        Args:
            random_state (int): The seed for all random operations.
        """
        self.data_manager = None
        self.random_state = random_state
        
        self.study = None
        self.best_params = None
        self.best_iteration = None
        self.model = None
        
        # Base params are still useful for both training and loading
        self.base_params: Dict[str, Any] = {
            'objective': 'regression_l1',
            'metric': 'l1',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }
        print("LightGBM trainer initialized.")

    @classmethod
    def from_training_data(cls, path: str, test_size: float = 0.1,
                           val_size: float = 0.15, random_state: int = 42):
        """
        Class method to create and initialize a LightGBM instance
        with data for a full training pipeline.
        Returns:
            LightGBM: A class instance loaded with data, ready for training.
        """
        # Create the instance using the standard __init__
        instance = cls(random_state=random_state)
        print("Initializing DataManager for training...")
        instance.data_manager = DataManager(
            path=path,
            model_type="lgbm",  # Hardcoded for this class
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        print("DataManager loaded. Trainer is ready for pipeline.")
        return instance


    def _objective(self, trial: optuna.Trial) -> float:
        """Internal objective function for Optuna tuning."""
        if self.data_manager is None:
            raise ValueError("Cannot run objective. DataManager not initialized. Use `from_training_data()` to initialize.")
        
        params = {
            **self.base_params,
            'n_estimators': 6000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', -1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            self.data_manager.X_train, self.data_manager.y_train,
            eval_set=[(self.data_manager.X_val, self.data_manager.y_val)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        trial.set_user_attr('best_iteration', model.best_iteration_)
        log_preds = model.predict(self.data_manager.X_val)
        final_y_val = np.expm1(self.data_manager.y_val)
        final_preds = np.expm1(log_preds)
        dollar_mae = mean_absolute_error(final_y_val, final_preds)
        
        if trial.number in [1, 2]:
            print(f"Plotting L1 loss for Trial {trial.number}...")
            l1_values = model.evals_result_['valid_0']['l1']
            plt.figure(figsize=(10, 5))
            plt.plot(l1_values)
            plt.title(f'Trial {trial.number} - Validation L1 Loss over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('L1 Loss')
            plt.grid()
            plt.show()
        
        return dollar_mae

    def tune_hyperparameters(self, n_trials: int = 50):
        """Runs the Optuna hyperparameter tuning process."""
        if self.data_manager is None:
            raise ValueError("Cannot tune. DataManager not initialized. Use `from_training_data()` to initialize.")

        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_trial.params
        self.best_iteration = self.study.best_trial.user_attrs['best_iteration']
        
        print("\n--- Tuning Complete ---")
        print(f"Best Validation MAE: ${self.study.best_trial.value:,.2f}")
        print(f"Best iteration found: {self.best_iteration}")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"    {key}: {value}")


    def train_final_model(self, save_path: Optional[str] = "final_lgbm_model.txt"):
        """
        Trains the final model on (train + validation) data.

        Args:
            save_path (Optional[str]): Path to save the final model.
                                       If None, model is not saved.
        """
        if self.data_manager is None:
            raise ValueError("Cannot train. DataManager not initialized. Use `from_training_data()` to initialize.")
        if self.best_params is None or self.best_iteration is None:
            raise ValueError("Must run `tune_hyperparameters()` before training.")

        print("\nTraining final model on (Train + Val) data...")
        final_params = {**self.base_params, **self.best_params, 'n_estimators': self.best_iteration}
        self.model = lgb.LGBMRegressor(**final_params)
        self.model.fit(self.data_manager.X_train_val, self.data_manager.y_train_val)
        print("Successfully fit the final model!")

        if save_path:
            try:
                print(f"Saving final model to {save_path}...")
                self.model.booster_.save_model(save_path)
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving model: {e}")

    def evaluate_model(self):
        """Evaluates the final model on the Train-Val and Test datasets."""
        if self.data_manager is None:
            raise ValueError("Cannot evaluate. DataManager not initialized. Use `from_training_data()` to initialize.")
        if self.model is None:
            raise ValueError("Model has not been trained yet. Run `train_final_model()` first.")
        train_val_log_preds = self.model.predict(self.data_manager.X_train_val)
        calculate_scaled_MAE(self.data_manager.y_train_val, train_val_log_preds, test=False)
        final_log_preds = self.model.predict(self.data_manager.X_test)
        calculate_scaled_MAE(self.data_manager.y_test, final_log_preds, test=True)

    def plot_feature_importance(self, top_n: int = 15):
        """Plots the top N feature importances from the trained model."""
        if self.data_manager is None:
            raise ValueError("Cannot plot importance. DataManager not initialized. Use `from_training_data()` to initialize.")
        if self.model is None:
            raise ValueError("Model has not been trained yet. Run `train_final_model()` first.")

        print(f"\nPlotting top {top_n} feature importances...")
        importances = self.model.feature_importances_
        feature_names = self.data_manager.X_train_val.columns
        
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
        plt.title(f'Top {top_n} Feature Importances (from LightGBM)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def run_lgbm_pipeline(self, n_trials: int = 50, top_n_features: int = 15,
                          model_save_path = "final_lgbm_model.txt"):
        """
        Runs the complete end-to-end training and evaluation pipeline.

        Args:
            n_trials (int): The number of tuning trials to perform.
            top_n_features (int): The number of top features to plot.
            model_save_path (Optional[str]): Path to save the final model.
        """
        if self.data_manager is None:
            raise ValueError("Cannot run pipeline. DataManager not initialized. Use `from_training_data()` to initialize.")
            
        print("======== 🚀 STARTING LGBM PIPELINE 🚀 ========")
        print("\n--- STAGE 1: HYPERPARAMETER TUNING ---")
        self.tune_hyperparameters(n_trials=n_trials)
        print("\n--- STAGE 2: FINAL MODEL TRAINING ---")
        self.train_final_model(save_path=model_save_path)
        print("\n--- STAGE 3: MODEL EVALUATION ---")
        self.evaluate_model()
        print("\n--- STAGE 4: FEATURE IMPORTANCE ---")
        self.plot_feature_importance(top_n=top_n_features)
        print("\n======== ✅ LGBM PIPELINE COMPLETE ✅ ========")

    def predict(self, data_path: str, pretrained_model_path: str = None) -> float:
        """
        Loads data, preprocesses it, and returns a single scaled prediction.
        Args:
            data_path (str): File path to the CSV with the data point.

            pretrained_model_path (str): Path to a saved LightGBM
            model file. If provided, this model is used. If None,
            the model from the current session is used.
        Returns:
            float: The final, un-logged (scaled to dollars) prediction.
        """
        model_to_use = None
        if pretrained_model_path:
            print(f"Loading pretrained model from {pretrained_model_path}...")
            try:
                model_to_use = lgb.Booster(model_file=pretrained_model_path)
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
                    processed_df[col] = processed_df[col].astype('category')
            
            model_features = model_to_use.feature_name()
            processed_df = processed_df[model_features] # ensure correct ordering/subset
            log_pred = model_to_use.predict(processed_df)
            final_pred = np.expm1(log_pred)
            
            return float(final_pred[0])

        except FileNotFoundError:
            print(f"Error: File not found at {data_path}")
            raise
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise