import os
from utils import load_data
from utils import clean_data
from utils import data_shuffle
from utils import save_results


if __name__ == '__main__':

    data_class = 0  # xgj_MR=0 ; jz_CT=1
    option = 0  # 去除术后数据=0 ; 保留术后数据=1

    if data_class == 0:
        data_type = 'xgj_MR'
        file_path = 'data/膝关节MR2020-2022.xlsx'
    elif data_class == 1:
        data_type = 'jz_CT'
        file_path = 'data/颈椎非结构化.xls'

    file_names = ['train.source', 'test.source', 'val.source', 'train.target', 'test.target', 'val.target']

    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), file_path))
    data = clean_data(load_data(data_path), option)

    for c in ['class_1', 'class_2', 'class_3', 'class_4']:
        input_train, input_test, input_val, output_train, output_test, output_val = data_shuffle(data, c, data_class)
        file_paths = [os.path.join('data', data_type, c, f) for f in file_names]
        abs_paths = [os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), p)) for p in file_paths]
        for p in set([os.path.dirname(ap) for ap in abs_paths]):
            os.makedirs(p, exist_ok=True)
        results = [(input_train, abs_paths[0]),
                   (input_test, abs_paths[1]),
                   (input_val, abs_paths[2]),
                   (output_train, abs_paths[3]),
                   (output_test, abs_paths[4]),
                   (output_val, abs_paths[5])]
        for r in results:
            save_results(r[0], r[1])

    print('保存完成')