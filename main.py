import argparse

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
            train_lgbm()
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