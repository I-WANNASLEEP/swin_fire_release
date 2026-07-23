#!/usr/bin/env python3
"""
Generate train/val/test NPY datasets from raw TS-SatFire TIF files.
Uses official split files and AFBADatasetProcessor.
Usage:
    python scripts/generate_af_dataset.py -mode train -ts 10 -it 3
    python scripts/generate_af_dataset.py -mode val   -ts 10 -it 3
    python scripts/generate_af_dataset.py -mode test  -ts 10 -it 3
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.splits import load_event_ids
from satimg_dataset_processor.satimg_dataset_processor import AFBADatasetProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate active-fire dataset')
    parser.add_argument('-mode', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('-ts', type=int, default=10)
    parser.add_argument('-it', type=int, default=3)
    parser.add_argument('-uc', type=str, default='af')
    parser.add_argument('--data-path', type=str, default='/home/congwei/archive/ts-satfire')
    parser.add_argument('--output-root', type=str, default='/home/congwei/swin_fire_new/data/processed')
    args = parser.parse_args()

    mode = args.mode
    ts_length = args.ts
    interval = args.it
    usecase = args.uc
    data_path = args.data_path
    output_root = Path(args.output_root)

    if mode == 'train':
        event_ids = load_event_ids(Path(__file__).resolve().parents[1] / 'splits/train_event_ids.txt')
        save_dir = output_root / 'dataset_train'
        file_name = f'{usecase}_train_img_seqtoseq_alll_{ts_length}i_{interval}'
        label_name = f'{usecase}_train_label_seqtoseq_alll_{ts_length}i_{interval}'
    elif mode == 'val':
        event_ids = load_event_ids(Path(__file__).resolve().parents[1] / 'splits/validation_event_ids.txt')
        save_dir = output_root / 'dataset_val'
        file_name = f'{usecase}_val_img_seqtoseq_alll_{ts_length}i_{interval}'
        label_name = f'{usecase}_val_label_seqtoseq_alll_{ts_length}i_{interval}'
    else:
        event_ids = load_event_ids(Path(__file__).resolve().parents[1] / 'splits/test_event_ids.txt')
        save_dir = output_root / 'dataset_test'

    os.makedirs(save_dir, exist_ok=True)

    processor = AFBADatasetProcessor()

    if mode in ('train', 'val'):
        print(f'Generating {mode} set: {len(event_ids)} events -> {save_dir}')
        processor.dataset_generator_seqtoseq(
            mode=mode, usecase=usecase, data_path=data_path,
            locations=event_ids,
            file_name=file_name, label_name=label_name,
            save_path=str(save_dir),
            visualize=False,
            ts_length=ts_length, interval=interval,
            rs_idx=0, cs_idx=0, image_size=(256, 256),
        )
        print(f'Done: {save_dir}/{file_name}.npy')
    else:
        print(f'Generating test set: {len(event_ids)} events -> {save_dir}')
        for eid in event_ids:
            print(f'  Processing: {eid}')
            processor.dataset_generator_seqtoseq(
                mode='test', usecase=usecase, data_path=data_path,
                locations=[eid], visualize=False,
                file_name=f'{usecase}_{eid}_img_seqtoseql_{ts_length}i_{interval}.npy',
                label_name=f'{usecase}_{eid}_label_seqtoseql_{ts_length}i_{interval}.npy',
                save_path=str(save_dir),
                ts_length=ts_length, interval=interval,
                rs_idx=0, cs_idx=0, image_size=(256, 256),
            )
        print(f'Done: {len(event_ids)} test events generated in {save_dir}')
