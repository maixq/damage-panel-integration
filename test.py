from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import time
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detector Classs
class PanelDetector():
    def __init__(self, panel_config, panel_checkpoint, panel_threshold=0.9, model_device='cpu'):
        self.panel_model = None
        self.panel_classes = None
        self.panel_class_metadata = None
        self.panel_config = panel_config
        self.panel_checkpoint = panel_checkpoint
        self.panel_threshold = panel_threshold
        self.model_device = model_device
        self.initialise_panel()

    def initialise_panel(self):

        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.panel_config)
        cfg.MODEL.WEIGHTS = self.panel_checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.panel_threshold
        cfg.MODEL.DEVICE = self.model_device
        self.panel_model = DefaultPredictor(cfg)
        logger.info('Panel Model Loaded')
    
    def predict(self, img):
        '''
        Function that takes in a Numpy Array and country code and returns an Masked Image or None
        '''
        t1 = time.time()
        predictions = self.panel_model(img)
        t2 = time.time()
        logger.info('Panel Detection Model Inference took: {}'.format(str(t2 - t1)))

        instances = predictions["instances"].to("cpu")

        v = Visualizer(
        img_rgb=img,
        instance_mode=ColorMode.SEGMENTATION
        )
        
        out = v.draw_instance_predictions(
        instances
        )
            
        result = {
            'image' : out,
        }

        return result, instances

# Car Detection class
class CarDetector():
    def __init__(self, config, checkpoint, threshold=0.9, model_device='cpu'):
        self.model = None
        self.config = config
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.model_device = model_device
        self.initialise()

    def initialise(self):
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.config)
        cfg.MODEL.WEIGHTS = self.checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.DEVICE = self.model_device
        self.model = DefaultPredictor(cfg)
    
    def predict(self, img):
        start_time = time.time()
        predictions = self.model(img)

        # Get detection boxes
        boxes = predictions['instances'].pred_boxes
        masks = predictions['instances'].pred_masks

        # If no boxes are detected, return original image and None for mask
        if len(boxes) == 0:
            return Image.fromarray(img), None
        else:
            # Crop Image
            box = list(boxes)[0].detach().cpu().numpy()
            cropped = self.crop_object(img, box)
            
            # Crop segmentation mask
            mask = list(masks)[0].detach().cpu().numpy()
            cropped_mask = self.crop_object(mask, box)
            
            return cropped, cropped_mask

    def crop_object(self, image, box):
        image = Image.fromarray(image)
        crop_img = image.crop(
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            )
        return crop_img
    

class DamageDetector():
    def __init__(self, config, checkpoint, threshold=0.5, model_device='cpu'):
        self.model = None
        self.classes = None
        self.class_metadata = None
        self.config = config

        self.checkpoint = checkpoint
        self.threshold = threshold
        self.model_device = model_device
        self.initialise()

    def initialise(self):
        self.classes = {
            'dent': 0,
            'paint_deterioration': 1,
            'scratch': 2,
            'dirt': 3,
            'reflection': 4,
        }
        self.class_metadata = MetadataCatalog.get('_').set(
            thing_classes=list(self.classes.keys())
            )
        
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.config)
        cfg.MODEL.WEIGHTS = self.checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.DEVICE = self.model_device
        self.model = DefaultPredictor(cfg)
    
    def predict(self, img, filter_class=True):
        start_time = time.time()
        self.model.model.eval()
        predictions = self.model(img)
        
        instances = predictions["instances"].to("cpu")
        
        if filter_class:
            # Filter and remove dirt and reflection classses
            instances = instances[instances.pred_classes != self.classes["dirt"]]
            instances = instances[instances.pred_classes != self.classes["reflection"]]

        '''
        # Filter center points to make sure that it is in the segmentation mask
        for row in instances.pred_boxes.get_centers().numpy().astype('int'):
            print(mask[row[1], row[0]])
        '''
        # TODO: Figure out how to filter using numpy arrays

        v = Visualizer(
            img_rgb=img,
            metadata=self.class_metadata,
            instance_mode=ColorMode.SEGMENTATION
            )

        out = v.draw_instance_predictions(
            instances
            )
            
        result = {
            'image' : out,
            'scratch_count' : len(instances[instances.pred_classes == self.classes["scratch"]]),
            'paint_deterioration_count' : len(instances[instances.pred_classes == self.classes["paint_deterioration"]]),
            'dent_count' : len(instances[instances.pred_classes == self.classes["dent"]]),
        }

        return result, instances

