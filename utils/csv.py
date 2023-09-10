import csv

def writeCSV(csvFile, data):
    with open(csvFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)