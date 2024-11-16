import argparse
from ds_prepper import DSPrepper

class LSStarter:

    def __init__(self, dataset_path: str, output_path: str, h5_path: str):
        prepper = DSPrepper(dataset_path, output_path)
        prepper.load_and_process()
        print("Saved h5 file")
        prepper.convert_to_h5(h5_path)

def main():
    parser = argparse.ArgumentParser(description="Load, process, and convert a dataset to HDF5.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to the input dataset CSV file."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the processed output CSV file."
    )
    parser.add_argument(
        "--h5_path", 
        type=str, 
        required=True, 
        help="Path to save the HDF5 file."
    )

    args = parser.parse_args()

    LSStarter(args.dataset_path, args.output_path, args.h5_path)

if __name__ == "__main__":
    main()
