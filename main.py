import os
import argparse
import shutil

from solver import Solver


def main(config):
    # Train and sample the images
    # solver = Solver(config)
    if config.phase in ["train", "training"]:
        solver = Solver(config)
        solver.train()
    elif config.phase in ["test", "testing"]:
        solver = Solver(config)
        solver.test()
    elif config.phase in ["tune", "tuning"]:
        assert 'Model_Tuning' in config.model_path, "Please change path to 'Model_Tuning' folder"
        if os.path.exists(config.model_path):
            shutil.rmtree(config.model_path)
        solver = Solver(config)
        solver.tuning()
    else:
        print('Mode not implemented')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to load/save the trained model
    parser.add_argument("--model_path", type=str,
                        default="results/models_test/")
    parser.add_argument("--model_name", type=str, default="model.pth")

    # Device setup
    parser.add_argument("--device_ids", type=list, default=[0, 1]) # [0, 1, 2, 3])

    # Hyper-parameters
    parser.add_argument("--model_type", type=str, default="UNet")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="PReLU")
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--tuning_log_dir", type=str, default=None)


    parser.add_argument("--dataset_path", type=str, default="polar_stent_dataset/")  # Path to load the train/valid datasets

    parser.add_argument("--l1_weight", type=float, default=1.0)  # L1 loss function weight
    parser.add_argument("--l2_penalty", type=float, default=0.001)  # L2 penalty for L2 regularization
    parser.add_argument("--sample_weight", type=tuple, default=(1.0, 1.0))  # Sample weight

    parser.add_argument("--phase", type=str, default="train")  # Phase - train or test
    parser.add_argument("--num_epochs", type=int, default=500)  # Total number of epochs

    parser.add_argument("--num_img_ch", type=int, default=1)  # The number of input image channels
    parser.add_argument("--num_classes", type=int, default=2)  # The number of output labels (bg included)
    parser.add_argument("--patch_size", type=tuple, default=(512, 512))  # Patch size for data augmentation

    parser.add_argument("--batch_size", type=int, default=32)  # Batch size for mini-batch stochastic gradient descent
    parser.add_argument("--num_workers", type=int, default=20)  # The number of workers to generate the images (def:20)

    parser.add_argument("--lr_opt", type=dict, default={"policy": "plateau",
                                                        "init": 1e-3,  # Initial learning rate
                                                        "term": 1e-7,  # Terminating learning rate condition
                                                        "gamma": 0.1,  # Learning rate decay level
                                                        "step": 0,  # Plateau step
                                                        "step_size": 20})  # Plateau length

    parser.add_argument("--threshold", type=float, default=0.5)  # Prediction threshold

    parser.add_argument("--resnet_type", type=str, default='resnet18')
    parser.add_argument("--weight_path", type=str, default=None)

    config_ = parser.parse_args()
    print(config_)
    main(config_)
