#!/usr/bin/env python3
"""trtinfer.py: An inference script that takes in a tensorRT engine and feeds input messages from ROS
and publishes the detections as images"""

__author__      = "Naeem Iqbal"
__email__       = "naeem.iqbal@dfki.de"
# ROS imports
import rospy
from sensor_msgs.msg import Image
import os
# Further imports
import time
import numpy as np
import torchvision.transforms as T
import cv2
import tensorrt as trt
# from cuda import cuda, nvrtc
# import pycuda.autoinit
import pycuda.driver as  cuda

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
    def __init__(self, trt_engine_path, topic, output_topic, batch_size=1):
        rospy.init_node('trt_infer')
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
        # self.batch_size = batch_size
        # if self.engine.has_implicit_batch_dimension:
        #     assert self.batch_size <= self.engine.max_batch_size
        # else:
        #     # self.batch_size = self.engine.max_batch_size
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        self.sample_img = cv2.imread('./sample.png')
        self.setup_IO_binding()
        # initialize transforms
        # self.transforms = T.Resize((self.input_shape[1], self.input_shape[2]))
        self.detection = rospy.Publisher(output_topic, Image, queue_size=10)
        self.class_ids = {0: 'weeds', 1: 'maize'}
        rospy.Subscriber(topic, Image, self.image_callback)
        rospy.spin()

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
            if -1 in shape:
                shape[0] = 300
            # size = np.dtype(trt.nptype(dtype)).itemsize
            # for s in shape:
            #     size *= s
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding_name).name=='INPUT':
                self.input = HostDeviceMem(host_mem, device_mem)
                self.context.set_input_shape(binding_name, [j for j in shape])
            elif self.engine.get_tensor_mode(binding_name).name=='OUTPUT':
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.context.set_binding_shape(i, [j for j in shape])
            else:
                pass

        # print(self.context.all_binding_shapes_specified)
        self.stream = cuda.Stream()
        # if isinstance(err, cuda.CUresult):
        #     if err != cuda.CUresult.CUDA_SUCCESS:
        #         raise RuntimeError("Cuda Error: {}".format(err))
        # assert self.batch_size == 1
        # assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.bindings) > 0
        # outputs = []
        # bindings = []
        # # for shape, dtype in self.output_spec():
        # #     outputs.append(np.zeros(shape, dtype))
        # for binding in self.engine:
        #     # from pycuda.driver import cuda
        #     binding_idx = self.engine.get_binding_index(binding)
        #     name = self.engine.get_tensor_name(binding_idx)
        #     size = trt.volume(self.context.get_binding_shape(binding_idx))
        #     dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        #     if self.engine.binding_is_input(binding):
        #         self.sample_img = cv2.resize(self.sample_img, (512, 683)).astype(dtype)
        #         self.sample_img = np.transpose(self.sample_img, (2, 1, 0))
        #         input_buffer = np.ascontiguousarray(self.sample_img)
        #         input_memory = cuda.mem_alloc(self.sample_img.nbytes)
        #         bindings.append(int(input_memory))
        #     else:
        #         output_buffer = cuda.pagelocked_empty(size, dtype)
        #         output_memory = cuda.mem_alloc(output_buffer.nbytes)
        #         bindings.append(int(output_memory))
        # stream = cuda.Stream()
        # # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # # Run inference
        # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # # Transfer prediction output from the GPU.
        # cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # # Synchronize the stream
        # stream.synchronize()
        # size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
        # host_mem = cuda.pagelocked_empty(size, self.dtype)
        # device_mem = cuda.mem_alloc(host_mem.nbytes)
        # bindings.append(int(device_mem))
        # if self.engine.binding_is_input(binding):
        #     inputs.append(HostDeviceMem(host_mem, device_mem))
        # else:
        #     outputs.append(HostDeviceMem(host_mem, device_mem))

    def scale_boxes(self, boxes, current_size=(800, 1067), new_size=(1536, 2048)):
        x_factor = new_size[0] / current_size[0]
        y_factor = new_size[1] / current_size[1]
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
        t1 = time.time()
        self.in_tensor = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.in_tensor = cv2.resize(self.in_tensor, (self.input_shape[2], self.input_shape[1]))
        # orig = in_tensor.copy()
        # cv2.imshow('figure', self.in_tensor)
        # cv2.waitKey()
        # self.in_tensor = np.transpose(self.in_tensor, (2, 1, 0))
        # self.in_tensor = np.expand_dims(self.in_tensor, axis=0)
        self.in_tensor = self.in_tensor.astype(self.dtype)
        self.in_tensor.shape
        # Process I/O and execute the network.
        self.cuda_ctx.push()
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        err = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # err = self.context.execute_v2(bindings=self.bindings)
        self.cuda_ctx.pop()
        pred = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            # out.host = out.host.reshape()
            pred.append(out.host)
        self.stream.synchronize()
        # print('Inference done') 
        # conf_inds = np.where(pred[2] > 0.50)
        # filtered = {}
        # filtered[0] = pred[0][conf_inds]
        # filtered[1] = pred[1][conf_inds]
        # filtered[2] = pred[2][conf_inds]
        # filtered[3] = pred[3]
        # filtered[0] = self.scale_boxes(filtered[0],
        #                                 current_size=(self.input_shape[2],
        #                                               self.input_shape[1]),
        #                                 new_size=(msg.width,
        #                                           msg.height))


        # for obj in range(filtered[0].shape[0]):
        #     box = filtered[0][obj, :]
        #     if filtered[1][obj] == 0:
        #         color = (0, 0, 255)
        #     else:
        #         color = (255, 0, 0)
        #     cv2.rectangle(orig_image,
        #                 pt1=(int(box[0]), int(box[1])),
        #                 pt2=(int(box[2]), int(box[3])),
        #                 color=color,
        #                 thickness=2)
        #     cv2.putText(orig_image,
        #                 '{:.2f} {}'.format(filtered[2][obj], self.class_ids[filtered[1][obj]]),
        #                 org=(int(box[0]), int(box[1] - 10)),
        #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=0.5,
        #                 thickness=2,
        #                 color=color)
        # # cv2.imshow('prediction', orig)
        # # cv2.waitKey(0)
        # msg_frame = bridge.cv2_to_imgmsg(orig, encoding="rgb8")
        # msg_frame.header.stamp = rospy.Time.now()
        # detection.publish(msg_frame)
        t2 = time.time()
        rospy.loginfo('Single image inference time: {:.2f} seconds and FPS {:.3f}'.format(t2 - t1, 1 / (t2 - t1)))

if __name__ == '__main__':
    TrtInfer(
            trt_engine_path='/opt/git/ag_rosbag_for_testing/dino_int8.engine',
            topic='/camera/color/image_raw',
            output_topic='/camera/color/detections'
            )
