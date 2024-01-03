import json 

normal = [5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19,  22]
day = [5, 9, 10, 11, 12, 13, 15, 16, 17, 19]
night = [1, 2, 3, 4, 6, 7, 8, 22]
bad = [21, *list(range(23, 59))]
all = list(range(1, 59))
for i in [51, 52, 57, 58]:
    bad.remove(i)
    all.remove(i)

all_splits = [day, night, bad]
split_name = ['day', 'night', 'bad']
all_split_reverse = {}
for idx, split in enumerate(all_splits):
    all_split_reverse[split_name[idx]] = list(set(all) - set(split))

with open('split_reverse.json', 'w') as f:
    json.dump(all_split_reverse, f)
    

    

