import re
import sys
import csv

def extract_loss(file,csv_filename):
    with open(file, 'r') as f:
        lines = f.readlines()
        epochs = []
        losses = []
    # Regular expression pattern to match the loss and epoch
    pattern = r"metrics\[0\]:.*'train/loss': tensor\((\d+\.\d+).*'train/epoch': (\d+)"

    for line in lines:
        match = re.search(pattern, line)

        if match:
            losses.append(float(match.group(1)))
            epochs.append(int(match.group(2)))
    
    # Write the losses and epochs to a csv file
    with open(f'src/tests/RoPE/{csv_filename}', 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], losses[i]])

if __name__ == "__main__":
    file = sys.argv[1]
    csv_filename = sys.argv[2]
    extract_loss(file,csv_filename)