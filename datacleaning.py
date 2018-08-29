import json

filepath = '/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/page2art.json'
linekeep = '<p>'
output = "/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/test.txt"

#we need to clean the data by finding lines with <p>

def clean_data(jsonfile, lineskeep, outputfile):

    with open(jsonfile) as f:
        data = json.load(f)

    file = open(outputfile, "w+")
    
    count = 0
    for x in data:
        if lineskeep in x['test']:
            linetest = x['test']
            print(linetest)
            count+=1
            file.write(linetest)
            if count == 100:
                file.close()
                break
        else:
            print("nope")



clean_data(filepath, linekeep, output)


