from curses import panel
from detectors import PanelDetector, CarDetector, DamageDetector
from shapely import geometry
from detectron2.utils.visualizer import GenericMask
from yaml import parse
from dictionary import panel_classes, damage_d, old_damage_d
from PIL import Image
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='path to image folder')
args = parser.parse_args()

def create_panel_dic(panel_instances, panel_classes):
    panel_dic = {}
    panel_area = {}
    for i in range(len(panel_instances)):
        inst_mask = panel_instances[i].pred_masks.numpy()
        pred_class = panel_instances[i].pred_classes
        panel_name = [k for k, v in panel_classes.items() if v == pred_class]
        r, w, h = inst_mask.shape    
        panel_mask = GenericMask(inst_mask.reshape(w,h), w, h)
        polygon_area = []
        # print(len(panel_mask.polygons))
        for x in range(len(panel_mask.polygons)):
            polygon = panel_mask.polygons[x].reshape(-1,2)
            # print(polygon)
            if panel_name[0] not in panel_dic:
                panel_dic[panel_name[0]] = list()
                panel_area[panel_name[0]] = list()
            panel_dic[panel_name[0]].append(geometry.Polygon(polygon))
            polygon_area.append(geometry.Polygon(polygon).area)
        # print('max panel polygon area: ',np.max(polygon_area))

        panel_area[panel_name[0]].append(np.mean(polygon_area))

    return panel_dic, panel_area

def create_damage_dic(damage_instances, damage_d):
    damage_dic = {}
    area_dic = {}
    # list of damage centroids
    damage_centroids = damage_instances.pred_boxes.get_centers().cpu().numpy()
    damage_classes = damage_instances.pred_classes.to('cpu').numpy()
    for i, id in enumerate(damage_classes):
        class_name = [v for k,v in damage_d.items() if k==id]

        dam_mask = damage_instances[i].pred_masks.to('cpu').numpy()
        r, w, h = dam_mask.shape
        d_mask = GenericMask(dam_mask.reshape(w,h), w, h)
        polygons_area = []
        for x in range(len(d_mask.polygons)):
            d_polygon = d_mask.polygons[x].reshape(-1,2)
            # print(d_polygon)
            polygons_area.append(geometry.Polygon(d_polygon).area)

        # print('damage polygn area: ', polygons_area)
        
        if class_name[0] not in area_dic:
            area_dic[class_name[0]] = list()
            damage_dic[class_name[0]] = list()
        area_dic[class_name[0]].append(np.mean(list(polygons_area)))
        damage_dic[class_name[0]].append(geometry.Point(damage_centroids[i]))
    
    return damage_dic, area_dic

def locate_point(damage_dic, panel_dic, damage_area_dic, panel_area):
    summary_d = {}
    for (damage_k,damage_v) in damage_dic.items():
        for num,damage in enumerate(damage_v):
            # print(num)
            found = False
            if not found:
                for (panel_key, panel_val) in panel_dic.items():
                    if panel_key not in summary_d:
                        summary_d[panel_key] = dict()

                    for polygon in panel_val:
                        if polygon.contains(damage):
                            if damage_k not in summary_d[panel_key]:
                                summary_d[panel_key][damage_k] = list()
                            # print(panel_area[panel_key])
                            damage_area_percent = (damage_area_dic[damage_k][num]/panel_area[panel_key])*100
                            summary_d[panel_key][damage_k].append(damage_area_percent.tolist()) 
                            print('{} detected on {}!'.format(damage_k, panel_key))
                
                    if found:
                        break
            else:
                print('not located')
    return summary_d

def create_img_summary(img_path, panel_detector, damage_detector, car_detector, dam_d, pan_d):

    input_img = np.array(Image.open(img_path).convert('RGB'))
    # get cropped img 
    cropped, cropped_mask = car_detector.predict(input_img)

    # detect panels
    car_results, panel_instances = panel_detector.predict(np.array(cropped))

    # detect damage
    results, damage_instances = damage_detector.predict(np.array(cropped), filter_class=True)
    
    damage_dic, damage_area_dic = create_damage_dic(damage_instances, dam_d)

    panel_dic, panel_area = create_panel_dic(panel_instances, pan_d)

    Image.fromarray(results['image'].get_image()).save('{}.jpg'.format(img_path.split('/')[-1].split('.')[0]))

    Image.fromarray(car_results['image'].get_image()).save('{}_cropped.jpg'.format(img_path.split('/')[-1].split('.')[0]))

    damage_panel_d = locate_point(damage_dic, panel_dic, damage_area_dic, panel_area)

    return damage_panel_d

def car_damage_area(sum_d):
    car_summary = {}
    for img ,summary in sum_d.items():
        im_panel_damage = summary
        # print(im_panel_damage)
        for panel, damage_panel in im_panel_damage.items():
            # print(panel, damage_panel)
            for damage_name, area in damage_panel.items():
                # print(damage_name, np.sum(area), 'in', panel)
                if panel not in car_summary:
                    car_summary[panel] = np.sum(area)
                else:
                    car_summary[panel] += np.sum(area)
    return car_summary
            
 

if __name__ == "__main__":

    panel_detector = PanelDetector(
    panel_config='carro-ds-cv-carro_locus/license_plate_masking/configs/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml',
    panel_checkpoint='model.pth',
    )

    car_detector = CarDetector(
    config='damage_detection_data_batch_1/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
    checkpoint='car_detection_model/car_detector_1.pth'
    )
    

    # Create DamageDetector object
    damage_detector = DamageDetector(
    config='damage_detection_data_batch_1/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
    checkpoint='damage_model/damage_detector_1.pth',
    threshold=0.8
    )

    # read images in folder
    # folder = '/Users/maixueqiao/Downloads/Project/panel_detection/359777'
    folder = args.input

    summary_d = dict()

    for x, filename in enumerate(os.listdir(folder)):
        img_path = os.path.join(folder,filename)
        img_summary = create_img_summary(img_path, panel_detector, damage_detector, car_detector, old_damage_d, panel_classes)
        if img_summary:
            summary_d[filename] = img_summary
        else:
            summary_d[filename] = {}

    # print(summary_d)
    with open("output.json", "w") as f:
        json.dump(summary_d, f, indent=4)
    
    # creeate per car damage summary
    car_damage_sum = car_damage_area(summary_d)
    with open("car_damage_area.json", "w") as output_f:
        json.dump(car_damage_sum, output_f, indent=4)

 