import os

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'input')
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')


full_split = (os.path.join(input_dir, 'clicks_train.csv.gz'), os.path.join(input_dir, 'clicks_test.csv.gz'))
val_split = (os.path.join(cache_dir, 'clicks_val_train.csv.gz'), os.path.join(cache_dir, 'clicks_val_test.csv.gz'))

val_split_time = 950400000
test_split_time = 1123200000
