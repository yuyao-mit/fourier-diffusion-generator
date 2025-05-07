# config.py
vgg_path_1 = '/home/gridsan/yyao/pretrained/vgg16-397923af.pth'
vgg_path_2 = '/work2/10214/yu_yao/frontera/pretrained/vgg16-397923af.pth'

root_dir_1 = "/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deepfaker/dataset/raw"
root_dir_2 = "/work2/10214/yu_yao/frontera/deepfaker/dataset/raw"

Pixelshuffle=True
patch_size=240

num_workers=6

loss_weight = [0.7,0.4,0.2,0.1]

train_scale_list=[2,4,6,8,12]
test_scale_list=[2,4,6,8,12]
