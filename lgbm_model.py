import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from utils import DataManager, calculate_scaled_MAE

# Suppress Optuna's trial logging
#optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBM:
    """
    A wrapper class for training, tuning, and evaluating a LightGBM model.

    This class orchestrates the entire modeling pipeline after data loading:
    1. Hyperparameter tuning using Optuna.
    2. Training the final model on the full training + validation set.
    3.Evaluating the final model on the test set.
    4. Visualizing model results (e.g., feature importance).
    """

    def __init__(self, path: str = "data/processed/processed_v1.csv", load_data: bool = True,
                 test_size: float = 0.1, val_size: float = 0.15, random_state: int = 42):
        """
        Initializes the LightBGM trainer and its internal DataManager.

        Args:
            path (str): The file path to the dataset.
            test_size (float): The proportion of the dataset for the test split.
            val_size (float): The proportion of the training set for validation.
            random_state (int): The seed for all random operations.
        """
        print("Initializing DataManager...")
        self.data_manager: DataManager = DataManager(
            path=path,
            model_type="lgbm",
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        self.study = None
        self.best_params = None
        self.best_iteration = None
        self.model = None

        # Base parameters for all LightGBM models
        self.base_params = {
            'objective': 'regression_l1',  # MAE
            'metric': 'l1',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        print("LightBGM trainer initialized.")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Internal objective function used for Optuna hyperparameter tuning.
        Returns:
            float: The validation Mean Absolute Error (in dollars),
                   which Optuna will seek to minimize.
        """
        # Define the hyperparameter search space
        params = {
            **self.base_params,
            'n_estimators': 6000,  # will be pruned by early stopping
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', -1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        model = lgb.LGBMRegressor(**params)

        model.fit(
            self.data_manager.X_train,
            self.data_manager.y_train,
            eval_set=[(self.data_manager.X_val, self.data_manager.y_val)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        # Save the best iteration number found by early stopping
        trial.set_user_attr('best_iteration', model.best_iteration_)

        # Evaluate on validation set
        log_preds = model.predict(self.data_manager.X_val)
        final_y_val = np.expm1(self.data_manager.y_val)
        final_preds = np.expm1(log_preds)
        dollar_mae = mean_absolute_error(final_y_val, final_preds)

        # Plot L1 loss for the first few trials
        if trial.number in [1, 2, 3]:
            print(f"Plotting L1 loss for Trial {trial.number}...")
            l1_values = model.evals_result_['valid_0']['l1']
            plt.figure(figsize=(10, 5))
            plt.plot(l1_values)
            plt.title(f'Trial {trial.number} - Validation L1 Loss over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('L1 Loss')
            plt.grid()
            plt.savefig(f'trial_{trial.number}_l1_loss.png')
            plt.close()

        return dollar_mae

    def tune_hyperparameters(self, n_trials: int = 50):
        """
        Runs the Optuna hyperparameter tuning process over n_trials.
        """
        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Store the best results
        self.best_params = self.study.best_trial.params
        self.best_iteration = self.study.best_trial.user_attrs['best_iteration']

        print("="*20)
        print("Tuning complete")
        print(f"Best Validation MAE: ${self.study.best_trial.value:,.2f}")
        print(f"Best iteration found: {self.best_iteration}")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"    {key}: {value}")

    def train_final_model(self):
        """
        Trains the final LightGBM model using the best parameters found
        during tuning, using the combined (train + validation) dataset.
        """
        if self.best_params is None or self.best_iteration is None:
            raise ValueError("Must run `tune_hyperparameters()` before training the final model.")

        print("\nTraining final model on (Train + Val) data...")
        
        # Combine base params, best hyperparams, and best iteration
        final_params = {
            **self.base_params,
            **self.best_params,
            'n_estimators': self.best_iteration
        }

        self.model = lgb.LGBMRegressor(**final_params)

        # Train on the full (Train + Val) dataset
        self.model.fit(
            self.data_manager.X_train_val,
            self.data_manager.y_train_val
        )
        print("Successfully fit the final model!")

    def evaluate_model(self):
        """
        Evaluates the final model on the Train-Val and Test datasets
        using the imported `calculate_scaled_MAE` function for printing.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Run `train_final_model()` first.")

        print("="*20)
        print("Model Evaluation")
        
        # Evaluate on the (Train + Val) set
        train_val_log_preds = self.model.predict(self.data_manager.X_train_val)
        calculate_scaled_MAE(
            self.data_manager.y_train_val,
            train_val_log_preds,
            test=False
        )
        
        # Evaluate on the Test set
        final_log_preds = self.model.predict(self.data_manager.X_test)
        calculate_scaled_MAE(
            self.data_manager.y_test,
            final_log_preds,
            test=True
        )

    def plot_feature_importance(self, top_n: int = 15):
        """
        Plots the top N feature importances from the trained model.

        Args:
            top_n (int): The number of top features to display.
        """
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

        # Dynamically adjust figure height based on number of features
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

    def run_lgbm_pipeline(self, n_trials: int = 50, top_n_features: int = 15):
        """
        Runs the complete end-to-end training and evaluation pipeline.

        This method sequentially calls:
        1. tune_hyperparameters()
        2. train_final_model()
        3. evaluate_model()
        4. plot_feature_importance()

        Args:
            n_trials (int): The number of tuning trials to perform.
            top_n_features (int): The number of top features to plot.
        """
        print("======== STARTING LGBM PIPELINE ========")
        print("\n--- STAGE 1: HYPERPARAMETER TUNING ---")
        self.tune_hyperparameters(n_trials=n_trials)
        print("\n--- STAGE 2: FINAL MODEL TRAINING ---")
        self.train_final_model()
        print("\n--- STAGE 3: MODEL EVALUATION ---")
        self.evaluate_model()
        print("\n--- STAGE 4: FEATURE IMPORTANCE ---")
        self.plot_feature_importance(top_n=top_n_features)
        print("\n======== LGBM PIPELINE COMPLETE ========")