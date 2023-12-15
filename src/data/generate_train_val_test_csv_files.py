import os
import sys
import argparse
import random
import csv
from tqdm import tqdm

def generate_train_val_test_csv_files(full_csv_file_path:str, val_size=0.2, test_size=0.2, train_file='train.csv', val_file='val.csv', test_file='test.csv'):
    """
    Takes a full csv file and splits it into train, val and test csv files.
    """
    # Read the full csv file
    with open(full_csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        file_data = [row for row in reader]

    # Calculate the sizes for train, val, and test sets
    total_samples = len(file_data)
    val_samples = int(val_size * total_samples)
    test_samples = int(test_size * total_samples)
    train_samples = total_samples - val_samples - test_samples

    # Shuffle the file_data randomly
    random.shuffle(file_data)

    # Split the file_data into train, val, and test sets
    train_data = file_data[:train_samples]
    val_data = file_data[train_samples:train_samples + val_samples]
    test_data = file_data[train_samples + val_samples:]

    print(f'Number of images in  train set: {len(train_data)}')
    print(f'Number of images in  val set: {len(val_data)}')
    print(f'Number of images in test set : {len(test_data)}')

    print('Saving train.csv file...')
    # Write train data to train_file
    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)
    print(f'train.csv saved!')

    print('Saving val.csv file...')
    # Write val data to val_file
    with open(val_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(val_data)
    print(f'val.csv saved!')

    print('Saving test.csv file...')
    # Write test data to test_file
    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)
    print(f'test.csv saved!')


def generate_file_paths_csv(lst_file_path, output_dir, csv_file):
    # Read the .lst file
    with open(lst_file_path, 'r') as lst_file:
        lines = lst_file.readlines()

    csv_file = os.path.join(output_dir, csv_file)

    written_images = set()  # Track the written images to avoid duplicates

    for line in tqdm(lines):
        # Split the line into image id, target, and path
        line_splitted = line.strip().split()
        target = line_splitted[1]
        path = line_splitted[2]
        img_index = os.path.basename(path).split('_')[0]

        if img_index not in written_images:
            channels_paths = [line.strip().split()[2] for line in lines if os.path.basename(line.strip().split()[2]).split('_')[0] == img_index]
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow([img_index] + [target] + [channels_paths])
            written_images.add(img_index)

def main(args):

    # Argument parser
    parser = argparse.ArgumentParser()

    # ----- Data Retrieving ----- #

    parser.add_argument(
        "--directory",
        type=str,
        default='datasets/BBBC048/Ground_truth.lst',
        help="Path to the full .lst file",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default='datasets/BBBC048/',
        help="Path to save the .csv files",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of the training set",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Size of the validation set",
    )

    args = parser.parse_args()

    #Assuming you already have the list of image IDs, 'ids'
    train_file_path = args.save_path + 'train.csv'
    test_file_path = args.save_path + 'test.csv'
    val_file_path = args.save_path + 'val.csv'

    # Extract unique image ids
    print(f'Extracting image ids from {args.directory}')
    generate_file_paths_csv(lst_file_path=args.directory, output_dir=args.save_path, csv_file='full.csv')

    # Split and save
    print(f'Splitting and saving the full.csv file into train, val, and test csv files')
    csv_file_path = args.save_path + 'full.csv'
    generate_train_val_test_csv_files(full_csv_file_path=csv_file_path,val_size=args.val_size, test_size=args.test_size, train_file=train_file_path, val_file=val_file_path, test_file=test_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
