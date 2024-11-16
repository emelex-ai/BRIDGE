import argparse
import sys
import os
from ds_prepper import DSPrepper

module_path = os.environ["LATENT_SCOPE_PATH"]
if module_path not in sys.path:
    sys.path.append(module_path)

import latentscope as ls

class LSStarter:

    def __init__(self, dataset_path: str, output_path: str, h5_path: str, directory_name: str):
        prepper = DSPrepper(dataset_path, output_path, directory_name)
        dataset = prepper.load_and_process()
        print("Saved h5 file")
        prepper.ingest_file(dataset)
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

    parser.add_argument(
        "--directory_name",
        type=str,
        required=True,
        help="Name of latent scope resulting directory"
    )

    args = parser.parse_args()

    LSStarter(args.dataset_path, args.output_path, args.h5_path, args.directory_name)

if __name__ == "__main__":
    main()
