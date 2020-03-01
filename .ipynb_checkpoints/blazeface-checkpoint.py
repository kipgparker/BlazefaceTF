import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np

class BlazeFace:
    def __init__(self, anchors_dir, weights_dir, size = (128,128)):
        self.x_scale = size[1]
        self.w_scale = size[1]
        self.y_scale = size[0]
        self.h_scale = size[0]
        self.anchors = np.load(anchors_dir)

        self.blazeface = self.blazeFace()
        self.blazeface.load_weights(weights_dir)

    def blazeBlock(self,inp, out, x, stride = 1):#Blazeblock 
        channel_pad = out - inp
        if(stride == 2):
            y = DepthwiseConv2D((3, 3), strides=(2, 2), padding = "SAME")(x)
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            y = DepthwiseConv2D((3, 3), strides=(1, 1), padding = "SAME")(x)
        y = Conv2D(out,(1, 1), padding='SAME', strides=(1,1))(y)  
        if channel_pad > 0:
            x = concatenate([x, tf.zeros_like(x[:,:,:,:channel_pad])], axis=-1)
        z = add([x, y])
        z = Activation('relu')(z) 
        return z

    def backbone1(self,x):
        x = Conv2D(24, (5, 5), strides=(2, 2), padding = "SAME")(x)
        x = Activation('relu')(x) 
        x = self.blazeBlock(24, 24, x)
        x = self.blazeBlock(24, 28, x)
        x = self.blazeBlock(28, 32, x, 2)
        x = self.blazeBlock(32, 36, x)
        x = self.blazeBlock(36, 42, x)
        x = self.blazeBlock(42, 48, x, 2)
        x = self.blazeBlock(48, 56, x)
        x = self.blazeBlock(56, 64, x)
        x = self.blazeBlock(64, 72, x)
        x = self.blazeBlock(72, 80, x)
        x = self.blazeBlock(80, 88, x)
        return x

    def backbone2(self,x):
        x = self.blazeBlock(88, 96, x, 2)
        x = self.blazeBlock(96, 96, x)
        x = self.blazeBlock(96, 96, x)
        x = self.blazeBlock(96, 96, x)
        x = self.blazeBlock(96, 96, x)
        return x

    def blazeFace(self):
        inputs = Input(shape=(128,128,3))
        x = self.backbone1(inputs)
        h = self.backbone2(x)

        c1 = Conv2D(2, (1, 1), name = 'classifier_8')(x)
        c1 = Reshape((512,1))(c1)
        c2 = Conv2D(6, (1, 1), name = 'classifier_16')(h)
        c2 = Reshape((384,1))(c2)
        c = concatenate([c1,c2], axis = 1)

        r1 = Conv2D(32, (1, 1), name = 'regressor_8')(x)
        r1 = Reshape((512,16))(r1)
        r2 = Conv2D(96, (1, 1), name = 'regressor_16')(h)
        r2 = Reshape((384,16))(r2)
        r = concatenate([r1,r2], axis = 1)
    
        #I believe the standard ordering returns [r,c] but due to the ordering of the weights, it needs to be this way round
        model = Model(inputs=[inputs], outputs=[c, r])
        return model
    
    def preprocess(self, x):
        return np.array(x, dtype = np.float32) / 127.5 - 1.0
    
    def predict(self, x, max_faces=10, iou_threshold = 0.2, score_threshold = 0.01, soft_nms_sigma=0.2):
        
        x = self.preprocess(x)
        raw_scores, raw_boxes = self.blazeface.predict(x)

        boxes = np.empty_like(raw_boxes)
        x_center = raw_boxes[..., 0] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.x_scale * self.anchors[:, 3] + self.anchors[:, 1]
        w = raw_boxes[..., 2] / self.w_scale * self.anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * self.anchors[:, 3]
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        
        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y
            
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                tf.squeeze(boxes[:,:,:4]),
                tf.squeeze(raw_scores),
                max_faces,
                iou_threshold = iou_threshold,
                score_threshold = score_threshold,
                soft_nms_sigma = soft_nms_sigma
                )
        
        selected_boxes = tf.gather(boxes, selected_indices)
        
        return selected_indices


