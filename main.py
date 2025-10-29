import argparse
from lgbm_model import LightGBM
#from catboost_model import CatBoost


def parse_arguments():
    parser = argparse.ArgumentParser(description="Choose mode")

    mode_group = parser.add_mutually_exclusive_group(required=True) # ensure only one is used
    mode_group.add_argument("--training", action="store_true", help="Train a new model")
    mode_group.add_argument("--inference", action="store_true", help="Use a trained model for inference")

    parser.add_argument("--data_point", type=str, help="Path to data point for inference")
    parser.add_argument("--model", type=str, help="Model to be trained - default: lgbm",
                        choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument("--inference_model_path", type=str, help="Path to a saved model for inference", default="final_lgbm_model.txt")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.training:
        if args.model == "lgbm":
            print("Training LGBM model...")
            trainer = LightGBM.from_training_data(path="data/processed/processed_v1.csv", test_size=0.1, val_size=0.15)
            trainer.run_lgbm_pipeline(n_trials=20, top_n_features=15, model_save_path="final_lgbm_model.txt")

        elif args.model == "catboost":
            print("Training CatBoost model...")
            #train_catboost()
    elif args.inference:
        if args.data_point:
            if args.model == "lgbm":
                print("Using LGBM model for inference...")
                inference_client = LightGBM()
                try:
                    prediction = inference_client.predict(
                        data_path=args.data_point,
                        pretrained_model_path=args.inference_model_path
                    )
                    print("="*20)
                    print(f"Predicted Sales Price: ${prediction:,.2f}")

                except Exception as e:
                    print(f"\nAn error occurred during prediction: {e}")

            elif args.model == "catboost":
                print("Using CatBoost model for inference...")
                #inference_catboost(args.data_point)
        else:
            print("Please provide a data point for inference.")
    else:
        print("Please choose either training or inference mode.")
    


if __name__ == "__main__":
    main()