you nee to download the training dataset from http://images.cocodataset.org/zips/train2017.zip
and put it in the folder "src/models/dataset/coco/". and unzip the file, the file name should be 'train2017'

you nee to download the validation dataset from http://images.cocodataset.org/zips/val2017.zip
and put it in the folder "src/models/dataset/coco/". and unzip the file, the file name should be 'val2017'

you nee to download the annotations from http://images.cocodataset.org/annotations/annotations_trainval2017.zip
and put it in the folder "src/models/dataset/coco/". and unzip the file, the file name should be 'annotations'


finally the folder should be like this:

src/models/dataset/coco/
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
├── val2017
│   ├── 000000000139.jpg
│   ├── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── ...

