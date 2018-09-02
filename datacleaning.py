import json
import re

filepath = '/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/espnarticles.json'
linekeep = '<p>'
output = "/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/bodytext.txt"
cleaned = "/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/cleanedespn.txt"

#we need to clean the data by finding lines with <p>

def clean_data(jsonfile, lineskeep, outputfile):

    with open(jsonfile) as f:
        data = json.load(f)

    file = open(outputfile, "w+")
    
    for x in data:
        if lineskeep in x['test']:
            linetest = x['test']
            file.write(linetest)


    file.close()


def removetags(file):
    with open(file) as f:
        lines = f.readlines()
    allwords = ''.join(lines)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', allwords)
    text_file = open(cleaned, "w")
    text_file.write(cleantext)
    text_file.close()


clean_data(filepath, linekeep, output)
cleaned = removetags(output)


