import json

filepath = '/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/espnarticles.json'
linekeep = '<p>'
output = "/Users/scottgleave/Downloads/Billsimmonsproject/BS-Robot/BS Spider/test.txt"

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

def IDtags(txtfile):
    with open(txtfile) as f:
        x = f.readlines()
    count = 0
    for y in x:
        for z in y:
            print(z)
        count +=1



    print('Done')
#clean_data(filepath, linekeep, output)
IDtags(output)
'''
more stuff to remove:
<font> tags
a target AREF

'''
