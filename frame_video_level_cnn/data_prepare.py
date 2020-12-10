import os
import json
import shutil


if __name__ == "__main__":
    input_root = 'frames'
    output_root = 'data'
    train_json = 'metadata_train.json'
    test_json = 'metadata_val.json'

    video_clips = os.listdir(input_root)

    with open(train_json, 'r') as fp:
        train_info = json.load(fp)
    with open(test_json, 'r') as fp:
        test_info = json.load(fp)

    train_txt = []
    test_txt = []

    invalid = []

    for vid in video_clips:
        if vid in train_info.keys():
            start = train_info[vid]['anomaly_start']
            end = train_info[vid]['anomaly_end']
            n_frames = train_info[vid]['num_frames']
            data_split = 'train'
        elif vid in test_info.keys():
            start = test_info[vid]['anomaly_start']
            end = test_info[vid]['anomaly_end']
            n_frames = test_info[vid]['num_frames']
            data_split = 'test'
        else:
            raise RuntimeError(f'invalid video clip {vid}')
        
        jpgs = os.listdir(os.path.join(input_root, vid))

        if len(jpgs) != n_frames:
            invalid.append(vid)
            print(f'{vid} has {len(jpgs)} frames, which is different from {n_frames}')
        else:
            jpgs.sort()

            for i in range(start-1):
                if not os.path.exists(os.path.join(output_root, f'{vid}-0')):
                    os.makedirs(os.path.join(output_root, f'{vid}-0'))
                shutil.copy(os.path.join(input_root, vid, jpgs[i]), os.path.join(output_root, f'{vid}-0'))
                if data_split == 'train':
                    train_txt.append(f'data/{vid}-0 normal\n')
                elif data_split == 'test':
                    test_txt.append(f'data/{vid}-0 normal\n')

            for i in range(start-1, end):
                if not os.path.exists(os.path.join(output_root, f'{vid}-1')):
                    os.makedirs(os.path.join(output_root, f'{vid}-1'))
                shutil.copy(os.path.join(input_root, vid, jpgs[i]), os.path.join(output_root, f'{vid}-1'))
                if data_split == 'train':
                    train_txt.append(f'data/{vid}-1 anomaly\n')
                elif data_split == 'test':
                    test_txt.append(f'data/{vid}-1 anomaly\n')

            for i in range(end, len(jpgs)):
                if not os.path.exists(os.path.join(output_root, f'{vid}-2')):
                    os.makedirs(os.path.join(output_root, f'{vid}-2'))
                shutil.copy(os.path.join(input_root, vid, jpgs[i]), os.path.join(output_root, f'{vid}-2'))
                if data_split == 'train':
                    train_txt.append(f'data/{vid}-2 normal\n')
                elif data_split == 'test':
                    test_txt.append(f'data/{vid}-2 normal\n')

    with open('train_list.txt', 'w') as fp:
        for line in train_txt:
            fp.write(line)
    print(f'train: {len(train_txt)}')
    
    with open('test_list.txt', 'w') as fp:
        for line in test_txt:
            fp.write(line)
    print(f'test:  {len(test_txt)}')
