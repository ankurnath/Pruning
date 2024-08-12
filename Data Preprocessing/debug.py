# Open the file in read mode
with open(f'../../data/snap_dataset/Twitter.txt', 'r') as file:
    # Read and print the first 10 lines
    for i in range(10):
        line = file.readline()
        if line == '':
            break  # Stop if there are less than 10 lines in the file
        print(line.strip())