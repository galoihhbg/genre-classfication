import csv

with open('dataset/movies_test.dat', 'r', encoding='utf-8') as file:
    data = file.read()

lines = data.replace('|',',').split('\n')
csv_data = []
for line in lines:
    parts = line.split('::')
    csv_data.append(parts)

header = ['ID', 'Title', 'Genre']

with open('dataset/csv_files/movies_test.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_data)