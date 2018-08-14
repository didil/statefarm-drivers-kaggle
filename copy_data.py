import os, shutil, pathlib

full_data_dir = '/home/didil/Downloads/statefarm-drivers/imgs/train'

base_dir = './my-imgs'
pathlib.Path(base_dir).mkdir(exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
pathlib.Path(train_dir).mkdir(exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
pathlib.Path(validation_dir).mkdir(exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
pathlib.Path(test_dir).mkdir(exist_ok=True)

dirs = {'train': train_dir, 'validation': validation_dir, 'test': test_dir}

classnames = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

for (dataset_type, start, end) in [('train', 0, 1300), ('validation', 1300, 1600), ('test', 1600, 1900)]:
    print('Copying ' + dataset_type + ' data')
    for classname in classnames:
        print('Copying ' + classname + ' data')
        pathlib.Path(os.path.join(dirs[dataset_type], classname)).mkdir(exist_ok=True)
        file_names = os.listdir(os.path.join(full_data_dir, classname))[start:end]
        for file_name in file_names:
            src = os.path.join(full_data_dir, classname, file_name)
            dst = os.path.join(dirs[dataset_type], classname, file_name)
            shutil.copyfile(src, dst)
