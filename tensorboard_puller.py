import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars_from_event_file(event_file, tags):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    scalars_dict = {}
    for tag in tags:
        if tag in event_acc.Tags()['scalars']:
            scalars_dict[tag] = event_acc.Scalars(tag)
    return scalars_dict

def save_scalars_to_csv(scalars_dict, output_dir, base_name, full_path):
    for tag, scalars in scalars_dict.items():
        agent_name = tag.split('/')[0]
        agent_number = agent_name.split('agent')[-1].replace('_', '')
        # Construct the file name
        run_name = "_".join(full_path.split(os.sep)[-3:])
        agent_dir = os.path.join(output_dir, f"T{agent_number}")
        os.makedirs(agent_dir, exist_ok=True)
        output_file = os.path.join(agent_dir, f"{run_name}_{base_name}.csv")
        df = pd.DataFrame([(scalar.wall_time, scalar.step, scalar.value) for scalar in scalars], columns=['Wall Time', 'Step', 'Value'])
        df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

def find_event_files(root_dir):
    event_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'Detect_Component_Generated_Embeddings' in root:
            continue
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    return event_files

def find_all_iteration_avg_reward_tags(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    iteration_avg_reward_tags = [tag for tag in tags if 'iteration_avg_reward' in tag]
    return iteration_avg_reward_tags

def process_event_files(event_files, output_dir):
    for event_file in event_files:
        tags = find_all_iteration_avg_reward_tags(event_file)
        if tags:
            scalars_dict = extract_scalars_from_event_file(event_file, tags)
            base_name = os.path.basename(event_file).split('.')[0]  # Remove extension from event file
            full_path = os.path.dirname(event_file)  # Get the full path of the event file
            save_scalars_to_csv(scalars_dict, output_dir, base_name, full_path)
        else:
            print(f"No iteration_avg_reward tags found in file {event_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard scalars and save to CSV")
    parser.add_argument("root_dir", help="Root directory containing TensorBoard log directories")
    parser.add_argument("--output_dir", default=".", help="Directory to save CSV files")
    args = parser.parse_args()

    event_files = find_event_files(args.root_dir)
    if not event_files:
        print("No event files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    process_event_files(event_files, args.output_dir)

if __name__ == "__main__":
    main()
