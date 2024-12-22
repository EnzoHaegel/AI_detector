import os

data_dir = "data"
MAX_CHARS = 2048

for subfolder_name in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder_name)
    if os.path.isdir(subfolder_path):
        preprocess_subdir = os.path.join(data_dir, "preprocess_" + subfolder_name)
        if not os.path.exists(preprocess_subdir):
            os.makedirs(preprocess_subdir)
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                    lines = f_in.readlines()
                filtered_lines = [line for line in lines if len(line) > 120 and line.strip() and line.strip()[0].isalnum()]
                full_text = "".join(filtered_lines)
                truncated_text = full_text[:MAX_CHARS]
                output_file_name = "processed_" + file_name
                output_file_path = os.path.join(preprocess_subdir, output_file_name)
                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(truncated_text)
                print(f"Fichier {file_name} traité et sauvegardé dans {output_file_path}")
