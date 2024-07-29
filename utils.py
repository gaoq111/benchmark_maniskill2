import json

def write_out(filePath, data):
    with open(filePath, "w") as write_ot:
        type_file = filePath.split(".")[-1]
        if(type_file == "txt"):
            for var in data:
                if(isinstance(var, dict)):
                    for key in var:
                        write_ot.write(f"{key}: {var[key]}")
                        write_ot.write("\n")
                else:
                    write_ot.write(var)
                    write_ot.write("\n")
                
                write_ot.write("\n")
        else:
            for var in data:
                json.dump(var, write_ot)
                write_ot.write("\n")
def read_out(filePath):
    read_type = filePath.split(".")[-1]
    with open (filePath, "r") as fileIn:
        cases = []
        if(read_type == "jsonl"):
            for line in fileIn.readlines():
                cases.append(json.loads(line))
        elif(read_type == "txt"):
            for line in fileIn.readlines():
                cases.append(line.strip())
        elif(read_type == "json"):
            cases = json.load(fileIn)
    return cases