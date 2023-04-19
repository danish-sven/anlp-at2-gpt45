import codecs
import platform
import sys
import csv

def main():
    assert platform.python_version_tuple()[0] == '3', 'This program supports only python3'
    filename = sys.argv[1]

    # Prepare the data for the .csv file
    data = []
    with codecs.open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            cols = line.split('\t')
            if len(cols) > 4:
                orig = cols[4]
                corr = cols[4]
            if len(cols) > 5:
                corr = cols[5]
            data.append([orig, corr])

    return data

if __name__ == "__main__":
    # Collect data from the input file
    data = main()

    # Write the data to a .csv file
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['original', 'correct'])  # Write the column names
        csvwriter.writerows(data)  # Write the rows
