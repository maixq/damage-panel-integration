### Usage
The damage localisation project aims to provide a vehicle price estimation based on the number and types of the damages found on respective car panels. This estimates the cost of conditioning or repairing of vehicle per panel basis.

### Description
A car can have multiple images captured from different angles. For this project, car panels and damages are detected using panel and damage detection models using Meta's detectron2. There are 29 classes in the panel detection model and 5 types of damages in the damage detection model. The number of damages per panel will be aggregated to best estimate the price of a car. 
 

### Directory Structure
This is a sample directory structure. 

```
.
├── README.MD
├── car_detection_model
│   └── car_detector_1.pth
├── damage_detection_data_batch_1
│   ├── Base-PointRend-RCNN-FPN.yaml
│   ├── Base-RCNN-FPN.yaml
│   ├── pointrend_rcnn_R_101_FPN_3x_coco.yaml
│   ├── pointrend_rcnn_R_50_FPN_1x_cityscapes.yaml
│   ├── pointrend_rcnn_R_50_FPN_1x_coco.yaml
│   ├── pointrend_rcnn_R_50_FPN_3x_coco.yaml
├── damage_model
│   └── damage_detector_1.pth
├── final_panel.ipynb
├── images
│  └── sample_test.jpg
├── localise_damage.py
├── model.pth
├── output.json
└── detectors.py
``` 

| File          | Description |
| --- | --- |
| model.pth  | weights of panel detection model  |
| car_detector_1.pth  | weights of car detection model  |
|damage_detector_1.pth  | weights of car damage detection model  |
| detector.py  | return detector class  |
| localise_damage.py  | return car_damage_area.json with the percentage of damage in each panel |

### Instructions

1. Install detectron2 environment

2. Locate and calculate the percentage of damage per panel from different car angles

``` 
python localise_damage.py --input <path_to_image_folder>
```

