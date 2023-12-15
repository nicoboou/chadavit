'''Small script to generate a CSV file with 10% of the rows from the original CSV file. Used mainly to generate a low-shot dataset for testing purposes.'''

import pandas as pd
import random
import argparse


def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate a CSV file with 10% of the rows from the original CSV file.')

    parser.add_argument('--file_path', type=str, help='Path to the original CSV file.', required=True)
    parser.add_argument('--perc', type=float, help='Percentage of rows to be selected.', required=True)
    parser.add_argument('--output_file_path', type=str, help='Path to the output CSV file.', required=True)

    return parser.parse_args()

def main(args):

    # Load the data
    data = pd.read_csv(args.file_path)

    # Determine % of the total number of rows
    num_rows = len(data)
    percentage = int(num_rows * args.perc)

    # Randomly select % of the rows
    random_indices = random.sample(range(num_rows), percentage)
    subset_data = data.iloc[random_indices]

    # Save these rows into a new CSV file
    subset_data.to_csv(args.output_file_path, index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
