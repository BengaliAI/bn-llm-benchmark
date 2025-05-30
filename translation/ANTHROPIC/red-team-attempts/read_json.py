import json
data = []
jsonl_file_path = "./red_team_attempts.jsonl"



'''
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        data.append(json.loads(line))
'''
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    contents = json.loads(file.read())

print(contents[:1])
