## NextVPU YOLOX

## 1. Environment
    pytorch>=1.7.0, python>=3.6, Ubuntu/Windows, see more in 'requirements.txt'

## 2. Dataset

    download COCO:
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    
    unzip and put COCO dataset in following folders:
    /path/to/dataset/annotations/instances_train2017.json
    /path/to/dataset/annotations/instances_val2017.json
    /path/to/dataset/images/trainval2017/*.jpg
    
    change opt.data_dir = "/path/to/dataset" in 'config.py'

## 3. Model changes
    1)将原来的 focus 结构进行了等价替换;
    2)将原网络的 swish 替换为 relu;

## 4. Train

    NextVPU-Train: Train Customer Dataset(COCO format), 
        1) Organize the customer dataset into COCO dataset format;
        2) Modify Config.py: The main changes to the config are as follows;
            -->opt.backbone: such as "CSPDarknet-nano", "CSPDarknet-tiny", "CSPDarknet-s,m,l,x"
            -->opt.input_size: such as (416,416), (640,640)
            -->opt.test_size: such as (416,416), (640,640)
            -->opt.activation: 'relu', (We only support the conversion of the relu operator)
            -->opt.label_name: such as ['aeroplane', 'bicycle', 'bird', 'boat',
                                        'bottle', 'bus', 'car', 'cat', 'chair',
                                        'cow', 'diningtable', 'dog', 'horse',
                                        'motorbike', 'person', 'pottedplant',
                                        'sheep', 'sofa', 'train', 'tvmonitor']
            -->opt.train_ann = "/algdata03/xxxx/VOC_2007/voc_2007_train.json"  # COCO format train json
            -->opt.val_ann = "/algdata03/xxxx/VOC_2007/voc_2007_val.json"  # COCO format val json
            -->opt.data_dir = "/algdata03/xxxx/VOC_2007/JPEGImages/"  # trainval dataset images
            -->Make other changes, such as opt.basic_lr_per_img, opt.min_lr_ratio, opt.enable_mixup

## 5. Add Post-Processing Node
    1)将上述 train 过程训练所得 model_best.pth 先使用 deployment\onnx\convert\pth2onnx.py 导出为 ONNX 模型;
    2)将转换得到的 ONNX 模型转换为 NCNN 模型，并将其中网络的后处理部分拆除, 以yolox-nano为例, 将节点 937,938,924,968,969,955,999,1000,986 之后的后处理操作相关节点全部拆除(拆除前后模型见下面百度网盘链接)；
    3)为拆除之后的 NCNN 主干网络添加定义后处理节点的 postpp.prototxt, 并将添加的 postpp.prototxt 与模型一同放置(见百度网盘)；
    4)在Compiler转换过程中, 文件目录结构如下：
    -->input
        padded_img.jpg (举例: 对输入原始图片进行 pad 后的图片, pad 过程详见 data\data_augment.py 中 preproc() 函数, pad 过程所乘数值为 114.0)
        prepp_data1_1x416x416x3.bin (举例: 将pad后的图片进行flatten, 并保存为 .bin 格式作为输入)
    -->model
        model_fp32_with_Prepp_Postpp.param (添加前后处理的模型 param)
        model_fp32_with_Prepp_Postpp.bin (添加前后处理的模型 bin)
        prepp.prototxt (添加前处理节点)
        postpp.prototxt (添加后处理节点)
    -->model_best.pth (训练得到的 pytorch 模型)
    -->model_best_sim.onnx (将 pytorch 模型转换为 ONNX)
    -->拆除后处理节点前NCNN模型.param (和model中的模型进行比较, 查看针对前后处理的修改)
    -->拆除后处理节点前NCNN模型.bin
    
    百度网盘链接：
    链接：https://pan.baidu.com/s/1s8Xb7mUpQR43JSunNhEj5g 
    提取码：7dz0 
    (pytorch 模型(pth), 转换得到的 ONNX 模型及 NCNN 模型，还有添加后处理节点的 postpp.prototxt)
