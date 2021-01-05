import argparse
from random import shuffle


def parse_args():
    parser = argparse.ArgumentParser(description='Simple script to subsample a csv dataset')

    parser.add_argument('--csv_path', help='Dataset File', default='train.csv', type=str)

    parser.add_argument('--subsample_number', help='Number of Frames to Subsample to', default=5000, type=int)

    parser.add_argument('--out_path', help='output file', default='subsampled.csv', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    import pandas as pd

    args = parse_args()
    label_db = pd.read_csv(args.csv_path, names=['Frame', 'x1', 'y1', 'x2', 'y2', 'Label'])
    dfs = dict(tuple(label_db.groupby('Frame')))
    df_keys_list = list(dfs.keys())
    shuffle(df_keys_list)

    frames = df_keys_list[0:args.subsample_number]
    subsampled_df = dfs[df_keys_list[0]].copy()
    for frame in frames:
        train_df = subsampled_df.append(dfs[frame])

    subsampled_df.to_csv(args.out_path, index=False, header=False)
