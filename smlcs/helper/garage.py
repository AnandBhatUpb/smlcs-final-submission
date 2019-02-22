import csv
fields = ['first', 'second', 'third']
with open('../../dump/test.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)