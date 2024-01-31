#!/usr/bin/env python3

# ROS imports
import rospy
from sensor_msgs.msg import Image, CompressedImage

import os
# Further imports
import yaml
import time
import numpy as np
import logging
import torch 
import torchvision.transforms as T
import onnxruntime
import cv2
from cv_bridge import CvBridge, CvBridgeError
import imutils
import argparse


try:
    from yolo_nms import scale_boxes, non_max_suppression
except Exception as e:
    print("Import error")

class OnnxInfer():
    def __init__(self, model_path, topic, output_topic, confidence=0.5, io_binding=False):
        rospy.init_node('onnx_infer')
        self.model_path = model_path
        self.confidence = confidence
        if 'yolo' in model_path:
            self.yolo = True
        else:
            self.yolo = False
        self.exec_providers = onnxruntime.get_available_providers()
        self.exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
        # self.exec_provider = ['CPUExecutionProvider'] 
        # self.exec_provider = ['TensorrtExecutionProvider'] 
        self.session_options = onnxruntime.SessionOptions()
        # self.session_options.log_verbosity_level = 1
        print('[INFO] Using {} for inference'.format(self.exec_provider))
        self.session = onnxruntime.InferenceSession(
            self.model_path, 
            # sess_options=self.session_options, 
            providers=self.exec_provider)
        self.input = self.session.get_inputs()[0]
        # self.bridge = CvBridge()
        self.input_name = self.input.name
        self.input_shape = self.input.shape
        print('[INFO] Input image resolution {}'.format(self.input_shape))
        self.outputs = self.session.get_outputs()
        self.use_io_binding = io_binding
        if 'float16' in self.session.get_inputs()[0].type:
            self.dtype = np.float16
        elif 'uint8' in self.session.get_inputs()[0].type:
            self.dtype = np.uint8
        else:
            self.dtype = np.float32
        # self.dtype = np.uint8 if 'uint8' in self.session.get_inputs()[0].type else np.float32

        if self.use_io_binding:
            # bind model inputs and outputs
            self.io_binding = self.session.io_binding()
            #inputs 
            # self.in_tensor = np.zeros((3, self.input_shape[1], self.input_shape[2])).astype(np.float32) # for transformers
            self.in_tensor = np.zeros((3, self.input_shape[1], self.input_shape[2])).astype(self.dtype)
            if len(self.input_shape) == 4:
                self.in_tensor = np.expand_dims(self.in_tensor, axis=0)
            print(self.in_tensor.shape)
            X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(self.in_tensor, 'cuda', 0)
            self.io_binding.bind_input(
                self.input_name,
                device_type=X_ortvalue.device_name(),
                device_id=0,
                element_type=self.dtype,
                shape=X_ortvalue.shape(),
                buffer_ptr=X_ortvalue.data_ptr()
            )

            # outputs
            box_value = onnxruntime.OrtValue.ortvalue_from_shape_and_type([300, 4], np.float32, 'cuda', 0)
            class_value = onnxruntime.OrtValue.ortvalue_from_shape_and_type([300], np.int32, 'cuda', 0)
            conf_value = onnxruntime.OrtValue.ortvalue_from_shape_and_type([300], np.float32, 'cuda', 0)
            self.io_binding.bind_ortvalue_output(self.outputs[0].name, box_value)
            self.io_binding.bind_ortvalue_output(self.outputs[1].name, class_value)
            self.io_binding.bind_ortvalue_output(self.outputs[2].name, conf_value)
        else:
            pass

        # initialize transforms
        self.transforms = T.Resize((self.input_shape[1], self.input_shape[2]))
        self.bridge = CvBridge()
        self.detection = rospy.Publisher(output_topic, Image, queue_size=10)
        self.class_ids = {0: 'weeds', 1: 'maize'}
        self.orig_shape = None 
        if 'compressed' in topic:
            self.compressed = True
            rospy.Subscriber(topic, CompressedImage, self.image_callback)
        else:
            rospy.Subscriber(topic, Image, self.image_callback)
        rospy.spin()

    def scale_boxes(self, boxes, current_size=(800, 1067), new_size=(1536, 2048)):
        x_factor = new_size[0] / current_size[0]
        y_factor = new_size[1] / current_size[1]
        boxes[:, 0] = boxes[:, 0] * x_factor
        boxes[:, 2] = boxes[:, 2] * x_factor
        boxes[:, 1] = boxes[:, 1] * y_factor
        boxes[:, 3] = boxes[:, 3] * y_factor
        return boxes
    
    def _scale_box_array(self, box, source_dim=(512,512), orig_size=(760, 1280), padded=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        if padded:
            xtl, xbr = box[:, 0] * (orig_size[1] / source_dim[1]), \
                       box[:, 2] * (orig_size[1] / source_dim[1])
            ytl, ybr = box[:, 1] * (orig_size[0] / source_dim[0]), \
                       box[:, 3] * (orig_size[0] / source_dim[0])
        else:
            xtl, xbr = box[:, 0] * (orig_size[1] / self.input_size[1]), \
                       box[:, 2] * (orig_size[1] / self.input_size[1])
            ytl, ybr = box[:, 1] * orig_size[0] / self.input_size[0], \
                       box[:, 3] * orig_size[0] / self.input_size[0]
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))
        return np.concatenate((xtl, ytl, xbr, ybr, box[:, 4:6]), axis=1)
    
    def image_callback(self,msg):
        # print('[INFO] Received an image message . . ')
        t1 = time.time()
        if self.compressed:
            self.in_tensor = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.in_tensor = cv2.imread('coco.jpg')
            self.orig_shape = self.in_tensor.shape[0:2]
        else:
            self.in_tensor = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.orig_shape = [msg.height, msg.width]
        orig = self.in_tensor.copy()
        if len(self.input_shape) == 4:
            self.in_tensor = cv2.cvtColor(self.in_tensor.copy(), cv2.COLOR_RGB2BGR)
            orig = self.in_tensor.copy()
            img_0 = np.zeros((self.input_shape[2], self.input_shape[3], 3), dtype=np.float32)
            self.in_tensor = imutils.resize(self.in_tensor, width=self.input_shape[2])
            p_h, p_w = self.in_tensor.shape[0], self.in_tensor.shape[1]
            img_0[0:self.in_tensor.shape[0], 0:self.in_tensor.shape[1], :] = self.in_tensor 
            img_0 = img_0.astype(np.uint8)
            self.in_tensor = img_0.copy()
            # self.in_tensor = self.in_tensor[:,:,::-1]
            self.in_tensor = np.transpose(self.in_tensor, (2, 0, 1))
            self.in_tensor = self.in_tensor.astype(self.dtype) / 255.0
            self.in_tensor = np.expand_dims(self.in_tensor, axis=0)
        else:
            # cv2.imwrite('orig.jpg',self.in_tensor)
            self.in_tensor = cv2.resize(self.in_tensor, (self.input_shape[2], self.input_shape[1])).astype(self.dtype)
            # cv2.imwrite('input.jpg',self.in_tensor)
            # cv2.waitKey(0)
            self.in_tensor = np.transpose(self.in_tensor, (2, 0, 1))
        if self.use_io_binding:
            self.session.run_with_iobinding(self.io_binding)
            pred = self.io_binding.copy_outputs_to_cpu()
        else:
            pred = self.session.run(None, {self.input_name: self.in_tensor})
        # print('Inference done') 
        if self.yolo:
            pred = non_max_suppression(pred[0], conf_thres=self.confidence)
            # pred = self._scale_box_array(pred[0].cpu().numpy(),
            #                              source_dim=(288, 512),
            #                              orig_size=(720, 1280),
            #                              padded=True)
            pred = self._scale_box_array(pred[0].cpu().numpy(),
                                source_dim=(p_h, p_w),
                                orig_size=(msg.height, msg.width),
                                padded=True)    
            filtered = {}   
            filtered[0] = pred[:, 0:4]
            filtered[1] = pred[:, 5]
            filtered[2] = pred[:, 4]        
        else:
            conf_inds = np.where((pred[2] > self.confidence) & (pred[1] == 0))
            filtered = {}
            filtered[0] = pred[0][conf_inds]
            filtered[1] = pred[1][conf_inds]
            filtered[2] = pred[2][conf_inds]
            # filtered[3] = pred[3]
            filtered[0] = self.scale_boxes(filtered[0],
                                            current_size=(self.input_shape[1],
                                                        self.input_shape[2]),
                                            new_size=(self.orig_shape[0],
                                                      self.orig_shape[1]))


        for obj in range(filtered[0].shape[0]):
            box = filtered[0][obj, :]
            if filtered[1][obj] == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(orig,
                        pt1=(int(box[0]), int(box[1])),
                        pt2=(int(box[2]), int(box[3])),
                        color=color,
                        thickness=2)
            # cv2.putText(orig,
            #             '{:.2f} {}'.format(filtered[2][obj], self.class_ids[filtered[1][obj]]),
            #             org=(int(box[0]), int(box[1] - 10)),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.5,
            #             thickness=2,
            #             color=color)
        # cv2.imshow('prediction', orig)
        # cv2.waitKey(0)
        # cv2.destroyWindow('prediction')
        # msg_frame = self.bridge.cv2_to_imgmsg(orig, encoding="rgb8")
        pub_msg = Image()
        pub_msg.header.stamp = rospy.Time.now()
        pub_msg.height=self.orig_shape[0]
        pub_msg.width=self.orig_shape[1 ]
        pub_msg.encoding="bgr8"
        pub_msg.is_bigendian=False
        pub_msg.data = np.array(orig).tobytes()
        self.detection.publish(pub_msg)
        t2 = time.time()
        # rospy.loginfo('Single image inference time: {:.2f} seconds and FPS {:.3f}'.format(t2 - t1, 1 / (t2 - t1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='retina')
    args = parser.parse_args()
    OnnxInfer(
            #   model_path='/opt/fcos_R_50_1x.onnx',
            model_path='/models/retinanet_coco/1/model.onnx',
            # model_path='/dino_r50_4scale_12ep_512edge.onnx',
              topic='/ai_test_field/edge/hsos/sensors/zed2i/zed_node/rgb/image_rect_color/compressed',
              output_topic='/camera/color/detections', 
              confidence=args.conf,
              io_binding=False)
