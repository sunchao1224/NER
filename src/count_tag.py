num_per =0
num_org =0
num_loc =0
num_o =0
num_all = 0
num_misc =0
num_COURT=0
num_GOVERNMENT=0
num_LEGISLATION=0
for line in open('data/E-NER-Dataset-main/E-NER_train_5000.txt', 'r', encoding='utf-8'):
    line = line.rstrip()
    if line and line.split()[0] != "-DOCSTART-":
        tag = line.split()[-1]
        num_all+=1
        if "ORG" in tag:
            num_org+=1
        elif "PER" in tag:
            num_per+=1
            if tag != 'I-PER':
                print(tag)
        elif "LOC" in tag:
            num_loc+=1
        elif "MISC" in tag:
            num_misc+=1
        elif "COURT" in tag:
            num_COURT+=1
        elif "GOVERNMENT" in tag:
            num_GOVERNMENT+=1
        elif "LEGISLATION/ACT" in tag:
            num_LEGISLATION+=1
        elif "O" in tag:
            num_o+=1
print("num_all:", num_all)
print("num_per:", num_per, ":", num_per/num_all)
print("num_org:", num_org, ":", num_org/num_all)
print("num_loc:", num_loc, ":", num_loc/num_all)
print("num_misc:", num_misc, ":", num_misc/num_all)
print("num_government:", num_GOVERNMENT, ":", num_GOVERNMENT/num_all)
print("num_LEGISLATION:", num_LEGISLATION, ":", num_LEGISLATION/num_all)
print("num_COURT:", num_COURT, ":", num_COURT/num_all)
print("num_o:", num_o, ":", num_o/num_all)