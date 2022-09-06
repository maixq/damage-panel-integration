from test import PanelDetector, CarDetector, DamageDetector
from shapely import geometry
from detectron2.utils.visualizer import GenericMask
from dictionary import panel_classes, damage_d
from PIL import Image
import numpy as np

def create_panel_dic(panel_instances, panel_classes):
    panel_dic = {}

    for i in range(len(panel_instances)):
        inst_mask = panel_instances[i].pred_masks.numpy()
        pred_class = panel_instances[i].pred_classes
        panel_name = [k for k, v in panel_classes.items() if v == pred_class]
        r, w, h = inst_mask.shape    
        panel_mask = GenericMask(inst_mask.reshape(w,h), w, h)
        
        for x in range(len(panel_mask.polygons)):
            polygon = panel_mask.polygons[x].reshape(-1,2)
            if panel_name[0] not in panel_dic:
                panel_dic[panel_name[0]] = list()
            panel_dic[panel_name[0]].append(geometry.Polygon(polygon))

    return panel_dic

def create_damage_dic(damage_instances, damage_d):
    damage_dic = {}
    # list of damage centroids
    damage_centroids = damage_instances.pred_boxes.get_centers().cpu().numpy()

    damage_classes = damage_instances.pred_classes.to('cpu').numpy()
    for i, id in enumerate(damage_classes):
        class_name = [v for k,v in damage_d.items() if k==id]
        if class_name[0] not in damage_dic:
            damage_dic[class_name[0]] = list()
        damage_dic[class_name[0]].append(geometry.Point(damage_centroids[i]))

    return damage_dic

def locate_point(damage_dic, panel_dic):
    summary_d = {}
    for (damage_k,damage_v) in damage_dic.items():
        for damage in damage_v:
            found = False
            #print(damage)
            if not found:
                for (panel_key, panel_val) in panel_dic.items():
                    if panel_key not in summary_d:
                        summary_d[panel_key] = dict()
                    for polygon in panel_val:
                        if polygon.contains(damage):
                            if damage_k not in summary_d[panel_key]:
                                summary_d[panel_key][damage_k] = 1
                            else:
                               summary_d[panel_key][damage_k] += 1 
                            print('{} detected on {}!'.format(damage_k, panel_key))
                            found = True
                            break
                
                    if found:
                        break
            else:
                print('not located')
    return summary_d

if __name__ == "__main__":

    # Detector Objects
    panel_detector = PanelDetector(
        panel_config='carro-ds-cv-carro_locus/license_plate_masking/configs/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml',
        panel_checkpoint='model.pth',
        )

    # Create CarDetector object
    car_detector = CarDetector(     
    config='damage_detection_data_batch_1/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
    checkpoint='car_detection_model/car_detector_1.pth'
    )

    damage_detector = DamageDetector(
        config='damage_detection_data_batch_1/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
        checkpoint='damage_model/damage_detector_1.pth',
        )

        # Detect Image
    im_path = '/Users/maixueqiao/Downloads/Project/panel_detection/images/carro_4bUmlSxCVvsYBSoX.jpg'
    img = np.asarray(Image.open(im_path).convert('RGB'))

    # Panel Detection inference
    car_results, panel_instances = panel_detector.predict(img)
    out_im = Image.fromarray(car_results['image'].get_image())

    # # Car detection inference
    cropped, cropped_mask = car_detector.predict(img)
    # cropped.save('crop.png')

    # Damage Detection
    results, damage_instances = damage_detector.predict(np.array(img), filter_class=True)
    Image.fromarray(results['image'].get_image()).save('out8.jpg')

    damage_dic, panel_dic = create_damage_dic(damage_instances, damage_d), create_panel_dic(panel_instances, panel_classes)
    damage_panel_d = locate_point(damage_dic, panel_dic)
    print(damage_panel_d)
