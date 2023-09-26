import os

with open("results_paths.txt","r") as f:
	results_lines = f.readlines()

with open("targets_paths.txt","r") as f:
	target_lines = f.readlines()

results_dict = {}
for result_path in results_lines:
	stripped = result_path.split("_")[4:7]
	if stripped[0] in ['linear','nn','polynomial']:
		stripped = stripped[0:2]
	stripped = ''.join(stripped)
	results_dict[stripped] = result_path

targets_dict = {}
for target_path in target_lines:
	stripped = ''.join(target_path.split("/")[-1][0:-19].split('_')[0:-1])
	targets_dict[stripped] = target_path


for stripped in results_dict:
	if stripped in targets_dict.keys():
		print(f'working on: {stripped}')
		os.system(f'python3.9 tester.py -target_path="{(targets_dict[stripped])}" -result_path="{(results_dict[stripped])}"')

print('finished!')