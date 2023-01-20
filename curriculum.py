import json

y = {"agents" : []}
x = list(range(134))
y["agents"].append({"task_ids" : x, "max_steps" : 51200})


for _ in range(134):
    x.insert(0, x.pop(-1))
    y["agents"].append({"task_ids" : x, "max_steps" : 51200})

json_obj = json.dumps(y, indent=4)


with open("curriculum.json", "w") as f:
    f.write(json_obj)