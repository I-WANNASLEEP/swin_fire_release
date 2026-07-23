"""DEPRECATED_NOT_USED_IN_PAPER.

This legacy helper has an unsafe directory-order / first-ten-event test split and
does not represent the fixed partition used for the reproducible revision.  Keep
it only for historical inspection.  Use ``scripts/materialize_splits.py`` to
materialize the pinned event lists and the upstream generator at the pinned
revision to build train/validation/test arrays with a per-window manifest.
"""

import argparse
import pandas as pd
import os
from satimg_dataset_processor.satimg_dataset_processor import AFBADatasetProcessor, AFTestDatasetProcessor

val_ids = ['20568194', '20701026','20562846','20700973','24462610', '24462788', '24462753', '24103571', '21998313', '21751303', '22141596', '21999381', '22712904']

# 从2021年数据中选择测试集（使用与训练集/验证集不同的数据）
# 2021年测试数据目录中的所有ID
test_data_dir = '/home/congwei/swin_fire/data/raw_data'
test_ids = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

# 选择10个样本作为测试集
if len(test_ids) > 10:
    test_ids = test_ids[:10]

print(f"Test IDs: {test_ids}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    parser.add_argument('-uc', type=str, help='use case')
    args = parser.parse_args()
    ts_length = args.ts
    interval = args.it
    modes = args.mode
    usecase=args.uc
    
    if modes == 'test':
        locations = test_ids
        data_path = '/home/congwei/swin_fire/data/raw_data'
        save_path = '/home/congwei/swin_fire/data/dataset_test'
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        satimg_processor = AFBADatasetProcessor()
        for id in locations:
            print(f"Processing test sample: {id}")
            satimg_processor.dataset_generator_seqtoseq(mode='test', usecase=usecase, data_path=data_path, locations=[id], visualize=False, 
                                                        file_name=usecase+'_'+id+'_img_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy', 
                                                        label_name=usecase+'_'+id+'_label_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                        save_path=save_path, ts_length=ts_length, interval=interval, rs_idx=0, cs_idx=0, image_size=(256, 256))
