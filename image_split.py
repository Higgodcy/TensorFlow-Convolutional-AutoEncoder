import os
from PIL import Image
# ================================costomed zone===========================================
path_t = 'C:\\Users\DCY\MyData\dataset\data5\\train_raw\\'
path_v = 'C:\\Users\DCY\MyData\dataset\data5\\test_raw\\'
outpath_train = 'C:\\Users\DCY\MyData\dataset\data5\\train\\'
outpath_val = 'C:\\Users\DCY\MyData\dataset\data5\\test\\'
# 切割后的图片数量 row x col
size = 128
# ==================================================================================
def splitimage(path, size, outpath):
    img = Image.open(path)
    h, w= img.size
    i=1
    print('第%s张, Original image info: %sx%s, %s, %s' % (i,w, h, img.format, img.mode))
    print('开始处理图片切割, 请稍候...')

    s = os.path.split(path)
    if outpath == '':
        outpath = s[0]
        i+=1
    fn = s[1].split('.')
    basename = fn[0]
    ext = fn[-1]
    b = 0
    d = 0
    num = 0
    rowheight = h // size + 1
    rowheight=128
    h1 = (h - size)//rowheight
    h2 = int(h1)+1
    colwidth = w // size + 1
    colwidth=128
    w1 = (w - size)//colwidth
    w2 = int(w1)+1


    for r in range(w2):
        for c in range(h2):
            box = (b , d , b +  size, d +  size)
            img.crop(box).save(os.path.join(outpath, basename + '_' + str(num) + '.' + ext), ext)
            num+=1
            b = b + rowheight
            if b-(h2*rowheight)==0:
                b=0
        d = d + colwidth
        if d-(w2*colwidth)==0:
            d=0
    print('图片切割完毕，共生成 %s 张小图片。' % num)

# src = input('请输入图片文件路径：')


for img_name in os.listdir(path_t):
    img_path = path_t + img_name  # 每一个图片的地址
    splitimage(img_path, size, outpath_train)
print("已完成训练集")
# for img_name in os.listdir(path_v):
#     img_path = path_v + img_name  # 每一个图片的地址
#     splitimage(img_path, size, outpath_val)
# print("已完成测试集")

