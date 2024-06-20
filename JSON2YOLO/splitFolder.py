# from ultralytics.data.converter import convert_coco

# convert_coco(labels_dir="/data/macaque/v1/annotations/", use_keypoints=True)
import os
import cv2
from shutil import copy

# label_dir = 'coco_converted/labels/train'
# image_dir = '/data/macaque/v1/images'
# save_dir = 'coco_converted/images/train'

# image_dict = {}
# for img_name in os.listdir(image_dir):
#     image_dict[img_name[:-4]] = img_name

# #读取列表下的txt文件
# files = os.listdir(label_dir)
# for file in files:
#     img_path = image_dict[file[:-4]]
#     # 复制图片
#     # cv2.imwrite(os.path.join(save_dir,img_path),cv2.imread(os.path.join(image_dir,img_path)))
#     r_path = os.path.join(image_dir,img_path)
#     w_path = os.path.join(save_dir,img_path)
#     copy(r_path,w_path)
mylist = ["/workspace/coco_converted/images/train/01418849d54b3005.jpg",
"/workspace/coco_converted/images/train/0c58303601bfe7c9.jpg",
"/workspace/coco_converted/images/train/36733cdb5f4af01e.jpg",
"/workspace/coco_converted/images/train/369164910eb3151b.jpg",
"/workspace/coco_converted/images/train/41f41d2862619f01.jpg",
"/workspace/coco_converted/images/train/4dcf210b4377dc52.jpg",
"/workspace/coco_converted/images/train/62a9c3eb96333691.jpg",
"/workspace/coco_converted/images/train/638455b76ab67bf9.jpg",
"/workspace/coco_converted/images/train/6b5dd31eb57262b4.jpg",
"/workspace/coco_converted/images/train/7fb610998393ad71.jpg",
"/workspace/coco_converted/images/train/866a2628e2e34228.jpg",
"/workspace/coco_converted/images/train/8b9714142ca5aee4.jpg",
"/workspace/coco_converted/images/train/PRI_0384.jpg",
"/workspace/coco_converted/images/train/PRI_0674.jpg",
"/workspace/coco_converted/images/train/PRI_0795.jpg",
"/workspace/coco_converted/images/train/PRI_0838.jpg",
"/workspace/coco_converted/images/train/PRI_0973.jpg",
"/workspace/coco_converted/images/train/PRI_1244.jpg",
"/workspace/coco_converted/images/train/ZooA_0185.jpg",
"/workspace/coco_converted/images/train/ZooA_0267.jpg",
"/workspace/coco_converted/images/train/ZooA_0622.jpg",
"/workspace/coco_converted/images/train/ZooA_0659.jpg",
"/workspace/coco_converted/images/train/ZooA_0665.jpg",
"/workspace/coco_converted/images/train/ZooA_0670.jpg",
"/workspace/coco_converted/images/train/ZooA_0902.jpg",
"/workspace/coco_converted/images/train/ZooA_0938.jpg",
"/workspace/coco_converted/images/train/ZooA_1026.jpg",
"/workspace/coco_converted/images/train/ZooA_1181.jpg",
"/workspace/coco_converted/images/train/ZooA_1235.jpg",
"/workspace/coco_converted/images/train/ZooA_1385.jpg",
"/workspace/coco_converted/images/train/ZooA_1607.jpg",
"/workspace/coco_converted/images/train/ZooA_3565.jpg",
"/workspace/coco_converted/images/train/ZooB_0106.jpg",
"/workspace/coco_converted/images/train/ZooB_0107.jpg",
"/workspace/coco_converted/images/train/ZooB_0180.jpg",
"/workspace/coco_converted/images/train/ZooB_0347.jpg",
"/workspace/coco_converted/images/train/ZooB_0905.jpg",
"/workspace/coco_converted/images/train/ZooB_0907.jpg",
"/workspace/coco_converted/images/train/ZooB_0909.jpg",
"/workspace/coco_converted/images/train/ZooB_0910.jpg",
"/workspace/coco_converted/images/train/ZooB_1012.jpg",
"/workspace/coco_converted/images/train/ZooB_1016.jpg",
"/workspace/coco_converted/images/train/ZooB_1310.jpg",
"/workspace/coco_converted/images/train/ZooC_0153.jpg",
"/workspace/coco_converted/images/train/ZooC_0372.jpg",
"/workspace/coco_converted/images/train/ZooC_0682.jpg",
"/workspace/coco_converted/images/train/ZooC_1395.jpg",
"/workspace/coco_converted/images/train/ZooC_1587.jpg",
"/workspace/coco_converted/images/train/ZooC_1777.jpg",
"/workspace/coco_converted/images/train/ZooC_1925.jpg",
"/workspace/coco_converted/images/train/ZooC_1929.jpg",
"/workspace/coco_converted/images/train/ZooC_1975.jpg",
"/workspace/coco_converted/images/train/ZooC_1987.jpg",
"/workspace/coco_converted/images/train/ZooC_2207.jpg",
"/workspace/coco_converted/images/train/ZooC_2243.jpg",
"/workspace/coco_converted/images/train/ZooC_2357.jpg",
"/workspace/coco_converted/images/train/ZooC_2405.jpg",
"/workspace/coco_converted/images/train/ZooD_0222.jpg",
"/workspace/coco_converted/images/train/ZooD_0312.jpg",
"/workspace/coco_converted/images/train/ZooD_1012.jpg",
"/workspace/coco_converted/images/train/ZooD_1085.jpg",
"/workspace/coco_converted/images/train/ZooD_1273.jpg",
"/workspace/coco_converted/images/train/ZooD_1290.jpg",
"/workspace/coco_converted/images/train/ZooD_1415.jpg",
"/workspace/coco_converted/images/train/ZooD_1567.jpg",
"/workspace/coco_converted/images/train/ZooD_2391.jpg",
"/workspace/coco_converted/images/train/a052f0a168c14a73.jpg",
"/workspace/coco_converted/images/train/a0793ec121187d78.jpg",
"/workspace/coco_converted/images/train/a685331f4401009f.jpg",
"/workspace/coco_converted/images/train/a812d7074cb295d6.jpg",
"/workspace/coco_converted/images/train/a85a6e0b67a9ea28.jpg",
"/workspace/coco_converted/images/train/adfea817fa1dbf75.jpg",
"/workspace/coco_converted/images/train/b3fe21ff28a2f8c4.jpg",
"/workspace/coco_converted/images/train/b718f8d46d9343b9.jpg",
"/workspace/coco_converted/images/train/bee32f5508e17eb5.jpg",
"/workspace/coco_converted/images/train/c7c7653c0ad720f1.jpg",
"/workspace/coco_converted/images/train/cc2d75ded92a6d11.jpg",
"/workspace/coco_converted/images/train/ceddc339f3501207.jpg",
"/workspace/coco_converted/images/train/d3846f0563128ee9.jpg",
"/workspace/coco_converted/images/train/d4e2e356a93763a2.jpg",
"/workspace/coco_converted/images/train/d9bc6bf54857465d.jpg",
"/workspace/coco_converted/images/train/e4616a487148ccba.jpg",
"/workspace/coco_converted/images/train/ec726be978c77215.jpg",
"/workspace/coco_converted/images/train/f128dc5aa590fd26.jpg",
"/workspace/coco_converted/images/train/f1f5bdc1ce0e9309.jpg",]

for i in mylist:
    #删除文件
    os.remove(i)