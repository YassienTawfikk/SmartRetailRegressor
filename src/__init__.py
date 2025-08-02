from src.__00__paths import data_dir_list, output_dir_list

# Iterate all over the two lists to create directories
for path in data_dir_list + output_dir_list:
    path.mkdir(parents=True, exist_ok=True)
