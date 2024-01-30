#!/usr/bin/env python3
"""trtinfer.py: An inference script that takes in a tensorRT engine and feeds input messages from ROS
and publishes the detections as images"""

__author__      = "Naeem Iqbal"
__email__       = "naeem.iqbal@dfki.de"

import os
# Further imports
import time
import numpy as np
import torchvision.transforms as T
import cv2
import tensorrt as trt
# from cuda import cuda, nvrtc
# import pycuda.autoinit
import pycuda.driver as cuda
from yolo_nms import scale_boxes, non_max_suppression
try:
    import rospy
except:
    print('ERROR! Rospy not imported. ROS inference will not work!')
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        # self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def _cuda_error_check(args):
    """CUDA error checking."""
    err, ret = args[0], args[1:]
    if isinstance(err, cuda.CUresult):
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
      raise RuntimeError("Unknown error type: {}".format(err))
    # Special case so that no unpacking is needed at call-site.
    if len(ret) == 1:
      return ret[0]
    return ret

class TrtInfer():
    def __init__(self, 
                 trt_engine_path, 
                 topic='None', 
                 output_topic='None', 
                 batch_size=1, 
                 mode='video'):
        cuda.init()
        device = cuda.Device(0)
        self.cuda_ctx = device.make_context()
        self.engine_path = trt_engine_path
        self.dtype = np.int8
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.batch_size = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        # self.sample_img = cv2.imread('./sample.png')
        self.setup_IO_binding()
        # initialize transforms
        # self.transforms = T.Resize((self.input_shape[1], self.input_shape[2]))
        if mode=='ros':
            # ROS imports
            from sensor_msgs.msg import Image, CompressedImage
            from cv_bridge import CvBridge, CvBridgeError
            rospy.init_node('trt_infer')
            self.detection = rospy.Publisher(output_topic, Image, queue_size=10)
            # self.class_ids = {0: 'weeds', 1: 'maize'}
            self.class_ids = {0: 'person'}
            self.pub_msg = Image()
            self.bridge = CvBridge()
            if 'compressed' in topic:
                self.compressed = True
                rospy.Subscriber(topic, CompressedImage, self.image_callback)
            else:
                rospy.Subscriber(topic, Image, self.image_callback)
            rospy.spin()
        elif mode=='video':
            self.infer_video(path='person2.mp4')

    def load_engine(self, engine_path):
        trt.init_libnvinfer_plugins(None, "")   
        assert os.path.exists(engine_path)
        print("Reading engine from file {}".format(engine_path))          
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        return engine

    def setup_IO_binding(self):
        # Setup I/O bindings
        self.outputs = []
        self.bindings = []
        self.allocations = []
        for binding_name, i in zip(self.engine, range(len(self.engine))):
            is_input = False
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            shape = self.engine.get_tensor_shape(binding_name)
            if self.engine.get_tensor_mode(binding_name).name=='INPUT':
                is_input = True
                self.batch_size = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #self.engine.max_batch_size #shape[0]
                self.input_shape = shape
            # if -1 in shape:
            #     shape[0] = 300
            # size = np.dtype(dtype).itemsize
            # for s in shape:
            #     size *= s
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            # Here the size argument is the length of the flattened array and not the byte size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding_name).name=='INPUT':
                self.input = HostDeviceMem(host_mem, device_mem)
                # TODO Deprecated warning remove in future
                self.context.set_binding_shape(i, [j for j in shape])
                # self.context.set_input_shape(binding_name, [j for j in shape])
            elif self.engine.get_tensor_mode(binding_name).name=='OUTPUT':
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.context.set_binding_shape(i, [j for j in shape])
                # TODO Deprecated warning remove in future
                # self.context.set_input_shape(binding_name, [j for j in shape])
                self.output_shape = shape
            else:
                pass

        # print(self.context.all_binding_shapes_specified)
        self.stream = cuda.Stream()
        assert len(self.outputs) > 0
        assert len(self.bindings) > 0

    def infer_video(self, path):
        video_handle = cv2.VideoCapture(path)
        io_time = 0
        inference_time = 0
        post_process = 0
        total_frames = int(video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        if (video_handle.isOpened()== False): 
            print("Error opening video stream or file")
        while(frame_count <= total_frames):
            t1 = time.time()
            # Capture frame-by-frame
            ret, self.in_tensor = video_handle.read()
            frame_count +=1
            if ret:
                t1 = time.time()
                orig = self.in_tensor.copy()
                self.in_tensor = cv2.resize(self.in_tensor, (self.input_shape[3], self.input_shape[2])) 
                # resized = self.in_tensor.copy()
                # cv2.imwrite('sample.jpg', self.in_tensor)
                # cv2.imshow('resized-input', self.in_tensor)
                # cv2.waitKey(0)
                self.in_tensor = np.transpose(self.in_tensor, (2, 0, 1))    # HWC to CHW
                self.in_tensor = np.expand_dims(self.in_tensor, axis=0) # add batch dimension
                self.in_tensor = self.in_tensor / 255.0   # normalize for yolo
                self.in_tensor = np.ascontiguousarray(self.in_tensor.astype(np.dtype(np.float32))).ravel()
                # NOTE! Copy current image to the host buffer i.e. self.input.host
                np.copyto(self.input.host, self.in_tensor)
                # Process I/O and execute the network.
                # self.cuda_ctx.push()
                # Copy current image from host to device memory ----> part of IO operations
                cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
                t2 = time.time()
                io_time = round((t2-t1)*1000, 2)
                err = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                # err = self.context.execute_v2(bindings=self.bindings)
                # self.cuda_ctx.pop()
                pred = []
                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
                    # out.host = out.host.reshape()
                    pred.append(out.host.reshape(self.output_shape))
                self.stream.synchronize()
                t3 = time.time()
                inference_time=round((t3-t2)*1000, 2)
                pred = non_max_suppression(pred[0], conf_thres=0.4)
                filtered = pred[0].cpu()
                if len(pred[0]) != 0:
                    filtered = self.scale_boxes(filtered,
                                                current_size=(self.input_shape[2],
                                                              self.input_shape[3]),
                                                new_size=(720, 1280))
                    for obj in range(filtered.shape[0]):    
                        box = filtered[obj, 0:4]
                        if filtered[obj, 5] == 0:
                            color = (0, 0, 255)
                        else:
                            color = (255, 0, 0)
                        cv2.rectangle(orig,
                                    pt1=(int(box[0]), int(box[1])),
                                    pt2=(int(box[2]), int(box[3])),
                                    color=color,
                                    thickness=2)
                        # cv2.putText(orig_image,
                        #             '{:.2f} {}'.format(filtered[2][obj], self.class_ids[filtered[1][obj]]),
                        #             org=(int(box[0]), int(box[1] - 10)),
                        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #             fontScale=0.5,
                        #             thickness=2,
                        #             color=color)
                    cv2.imshow('prediction', orig)
                    cv2.waitKey(0)
                else:
                    pass
                    # print('No predictions generated by the model')
                # msg_frame = bridge.cv2_to_imgmsg(orig, encoding="rgb8")
                # msg_frame.header.stamp = rospy.Time.now()
                # detection.publish(msg_frame)
                t4 = time.time()
            post_process = round((t4-t3)*1000, 2)
            print('IO loading: {} ms Inference: {} ms PostProcess: {} ms FPS {}'.format(
                io_time, 
                inference_time, 
                post_process,
                (1/((io_time+inference_time+post_process)/1000))))
        # When everything done, release the video capture object
        video_handle.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        # Gracefully pop all cuda contexts
        self.cuda_ctx.pop()

    def scale_boxes(self, boxes, current_size=(800, 1067), new_size=(1536, 2048)):
        y_factor = new_size[0] / current_size[0]
        x_factor = new_size[1] / current_size[1]
        boxes[:, 0] = boxes[:, 0] * x_factor
        boxes[:, 2] = boxes[:, 2] * x_factor
        boxes[:, 1] = boxes[:, 1] * y_factor
        boxes[:, 3] = boxes[:, 3] * y_factor
        return boxes

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def image_callback(self,msg):
        # print('[INFO] Received an image message . . ')
        # outputs = []
        # for shape, dtype in self.output_spec():
        #     outputs.append(np.zeros(shape, dtype))
        if self.compressed:
            self.in_tensor = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.in_tensor = cv2.imread('coco.jpg')
            self.orig_shape = self.in_tensor.shape[0:2]
        else:
            self.in_tensor = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.orig_shape = [msg.height, msg.width]
        orig = self.in_tensor.copy()
        self.in_tensor = cv2.resize(self.in_tensor, (self.input_shape[3], self.input_shape[2])) 
        # resized = self.in_tensor.copy()
        # cv2.imwrite('sample.jpg', self.in_tensor)
        # cv2.imshow('resized-input', self.in_tensor)
        # cv2.waitKey(0)
        self.in_tensor = np.transpose(self.in_tensor, (2, 0, 1))    # HWC to CHW
        self.in_tensor = np.expand_dims(self.in_tensor, axis=0) # add batch dimension
        self.in_tensor = self.in_tensor / 255.0   # normalize for yolo
        self.in_tensor = np.ascontiguousarray(self.in_tensor.astype(np.dtype(np.float32))).ravel()
        # NOTE! Copy current image to the host buffer i.e. self.input.host
        np.copyto(self.input.host, self.in_tensor)
        # Process I/O and execute the network.
        self.cuda_ctx.push()
        # Copy current image from host to device memory ----> part of IO operations
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        # t2 = time.time()
        # io_time = round((t2-t1)*1000, 2)
        err = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # err = self.context.execute_v2(bindings=self.bindings)
        self.cuda_ctx.pop()
        pred = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            # out.host = out.host.reshape()
            pred.append(out.host.reshape(self.output_shape))
        self.stream.synchronize()
        pred = non_max_suppression(pred[0], conf_thres=0.3)
        filtered = pred[0].cpu()
        if len(pred[0]) != 0:
            filtered = self.scale_boxes(filtered,
                                        current_size=(self.input_shape[2],
                                                      self.input_shape[3]),
                                        new_size=(self.orig_shape[0],
                                                  self.orig_shape[1]))
            for obj in range(filtered.shape[0]):    
                box = filtered[obj, 0:4]
                if filtered[obj, 5] == 0:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(orig,
                            pt1=(int(box[0]), int(box[1])),
                            pt2=(int(box[2]), int(box[3])),
                            color=color,
                            thickness=2)
                # cv2.putText(orig_image,
                #             '{:.2f} {}'.format(filtered[2][obj], self.class_ids[filtered[1][obj]]),
                #             org=(int(box[0]), int(box[1] - 10)),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5,
                #             thickness=2,
                #             color=color)
        if self.pub_msg is not None:
            self.pub_msg.header.stamp = rospy.Time.now()
            self.pub_msg.height=self.orig_shape[0]
            self.pub_msg.width=self.orig_shape[1]
            self.pub_msg.encoding="bgr8"
            self.pub_msg.is_bigendian=False
            self.pub_msg.data = np.array(orig).tobytes()
            self.detection.publish(self.pub_msg)
            # rospy.loginfo('Single image inference time: {:.2f} seconds and FPS {:.3f}'.format(t2 - t1, 1 / (t2 - t1)))
        

if __name__ == '__main__':
    TrtInfer(trt_engine_path='yolov5m_coco/1/yolov5m_coco_int8_v8.5.2.engine', 
             topic='/ai_test_field/edge/hsos/sensors/zed2i/zed_node/rgb/image_rect_color/compressed',
             output_topic='/camera/color/detections',
             mode='ros')
