# count the number of pixels in each affordance
import glob
import pandas as pd

path_list = glob.glob('./part-affordance-dataset/tools/*', recursive=True)

image_path_list_train = []
image_path_list_train_with_label = []
image_path_list_train_without_label = []
image_path_list_test = []


for path in path_list:
    i = glob.glob(path + '/*rgb.jpg')
    
    image_path_list_test += i[:50]
    image_path_list_train += i[50:]


for i, path in enumerate(image_path_list_train):
    if i%5 == 0:
        image_path_list_train_without_label.append(path)
    else:
        image_path_list_train_with_label.append(path)


class_path_list_train_with_label = []
class_path_list_test = []

for path in image_path_list_train_with_label:
    c = path[:-7] + 'label.mat'
    class_path_list_train_with_label.append(c)
    
for path in image_path_list_test:
    c = path[:-7] + 'label.mat'
    class_path_list_test.append(c)


df_train_with_label = pd.DataFrame({
    "image_path": image_path_list_train_with_label,
    "class_path": class_path_list_train_with_label},
    columns=["image_path", "class_path"]
)

df_train_without_label = pd.DataFrame({
    "image_path": image_path_list_train_without_label
})

df_test = pd.DataFrame({
    "image_path": image_path_list_test,
    "class_path": class_path_list_test},
    columns=["image_path", "class_path"]
)

df_train_with_label.to_csv('train_with_label_4to1.csv', index=None)
df_train_without_label.to_csv('train_without_label_4to1.csv', index=None)
df_test.to_csv('test.csv', index=None)