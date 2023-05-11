import os
import shutil
import numpy as np
import re

main_dir = "log_temp/debugging_dropout_preprocessed/"
final_path = ''
lowest_iterations = np.inf
i = 0

# Recursively search for files in subdirectories that match the pattern
for root, dirs, files in os.walk(main_dir):
    # Remove any directories that contain 'agent_0' or 'agent_13' from the search
    dirs[:] = [d for d in dirs if 'agent_0' not in os.path.join(root, d) and 'agent_13' not in os.path.join(root, d)]
    
    for subdir in dirs:
        subdir_path = os.path.join(root, subdir)

        # Look for all files in the subdirectory that match the given pattern
        for filename in os.listdir(subdir_path):
            if filename.startswith("train-log-") and filename.endswith(".txt"):
                file_path = os.path.join(subdir_path, filename)

                # Check if the file contains any content
                if os.path.getsize(file_path) > 0:
                    # Create backup only if it does not already exist
                    backup_path = os.path.join(subdir_path, "log_backup.txt")
                    if not os.path.exists(backup_path):
                        shutil.copyfile(file_path, backup_path)

                    # Find all instances of the phrase "task n / iteration x" in the file
                    with open(backup_path) as f:
                        content = f.read()
                        tasks_iterations = [task for task in content.split('\n') if 'task' in task and 'iteration' in task]
                        iterations = [int(re.findall(r"iteration\s(\d+)", task)[0]) for task in tasks_iterations]
                        print(iterations[-1], file_path)
                        if iterations[-1] > 300:
                            if iterations[-1] < lowest_iterations:
                                lowest_iterations = iterations[-1]
                                final_path = file_path
                        else:
                            i += 1
                else:
                    # Do nothing with the file
                    pass

print(f'{lowest_iterations} at path: {final_path}')
print(f'Dropping {i} results')

# Go through all the files again and keep text up to the occurrence of the phrase 'iteration x'
for root, dirs, files in os.walk(main_dir):
    for subdir in dirs:
        subdir_path = os.path.join(root, subdir)

        for filename in os.listdir(subdir_path):
            if filename.startswith("train-log-") and filename.endswith(".txt"):
                file_path = os.path.join(subdir_path, filename)
                backup_path = os.path.join(subdir_path, "log_backup.txt")

                # Check if the backup file exists
                if os.path.exists(backup_path):
                    with open(backup_path) as f:
                        content = f.read()
                        shortened_content = re.sub(rf"(.*?iteration {lowest_iterations}).*", r"\1", content, flags=re.DOTALL)
                        
                        # Write the shortened text as a new file called log_processed.txt in the directory the file was found
                        processed_path = os.path.join(subdir_path, "log_processed.txt")
                        with open(processed_path, "w") as f2:
                            f2.write(shortened_content)
                else:
                    # Do nothing with the file
                    pass
