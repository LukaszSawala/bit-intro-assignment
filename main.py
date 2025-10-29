import argparse
from lgbm_model import LightGBM
from catboost_model import CatBoost


def parse_arguments():
    parser = argparse.ArgumentParser(description="Choose mode")
    parser.add_argument("--training", action="store_true", help="Train a new model")
    parser.add_argument("--inference", action="store_true", help="Use a trained model for inference")
    parser.add_argument("--data_point", type=str, help="Path to data point for inference")
    parser.add_argument("--model", type=str, help="Model to be trained/used for inference - default: lgbm",
                        choices=["lgbm", "catboost"], default="lgbm")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.training:
        if args.model == "lgbm":
            print("Training LGBM model...")
            trainer = LightGBM(path="data/processed/processed_v1.csv",
            test_size=0.1, val_size=0.15)
            trainer.run_lgbm_pipeline(n_trials=50, top_n_features=15)

        elif args.model == "catboost":
            print("Training CatBoost model...")
            train_catboost()
    elif args.inference:
        if args.data_point:
            if args.model == "lgbm":
                print("Using LGBM model for inference...")
                inference_lgbm(args.data_point)
            elif args.model == "catboost":
                print("Using CatBoost model for inference...")
                inference_catboost(args.data_point)
        else:
            print("Please provide a data point for inference.")
    else:
        print("Please choose either training or inference mode.")
    


if __name__ == "__main__":
    main()