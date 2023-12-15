import os
import wandb
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default=0, help='version number')
    parser.add_argument('--resume', action='store_true', help='resume the run')
    parser.add_argument('--save_dir', type=str, default='.', help='directory where the training logs are saved')
    parser.add_argument('--confusion_matrix', action='store_true', help='whether to only log the confusion matrix')

    args = parser.parse_args()

    if args.confusion_matrix:

        wandb.init(id=args.version, project='bioFMv2', resume=args.resume)

        # check .png file begins wit val_confusion_matrix
        for file in os.listdir(args.save_dir):
            if file.startswith("val_confusion_matrix"):
                # check if has the same id as the run

                if file.split('_')[3].split('.')[0] == args.version:
                    wandb.log({"val_confusion_matrix": wandb.Image(os.path.join(args.save_dir, file))})
                    break

    else:
        # Find file that ends with ckpt
        for file in os.listdir(args.save_dir):
            if file.endswith(".ckpt"):
                ckpt_file = file
                # Get the name of the run without .ckpt
                run_name = ckpt_file.split('.')[0]
                break

        wandb.init(name=run_name, id=args.version, project='bioFMv2', resume=args.resume)

        # read the log file and log each line to wandb
        with open(args.save_dir + "/training_logs.txt", "r") as f:
            for line in f:
                # convert line to dict
                wandb.log(eval(line)) #allows to delete duplicates if any in terms of epoch

    wandb.finish()

if __name__ == "__main__":
    main()
