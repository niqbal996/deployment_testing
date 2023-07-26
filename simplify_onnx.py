import os
import sys
import logging
import onnx_graphsurgeon as gs
import onnx
import tensorrt as trt
import numpy as np
import common
from onnx import shape_inference
from cuda import cudart


logging.basicConfig(level=logging.INFO)
logging.getLogger("ModelHelper").setLevel(logging.INFO)
log = logging.getLogger("ModelHelper")

class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    """
    Implements the INT8 MinMax Calibrator.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    # def set_image_batcher(self, image_batcher: ImageBatcher):
    #     """
    #     Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
    #     to be defined.
    #     :param image_batcher: The ImageBatcher object
    #     """
    #     self.image_batcher = image_batcher
    #     self.size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
    #     self.batch_allocation = common.cuda_call(cudart.cudaMalloc(self.size))
    #     self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))

            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

def save(graph, output_path):
        """
        Save the ONNX model to the given location.
        :param output_path: Path pointing to the location where to write out the updated ONNX model.
        """
        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        output_path = os.path.realpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)
        log.info("Saved ONNX model to {}".format(output_path))
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        self.batch_size = None
        self.network = None
        self.parser = None
        self.engine_path = '/opt/data/retinanet_fp32.engine'

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

        # engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        # with open(self.engine_path, "wb") as f:
        #     log.info("Serializing engine to file: {:}".format(self.engine_path))
        #     f.write(engine_bytes)

    def create_engine(self, engine_path, precision, config_file, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=1):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16', 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision in ["fp16", "int8"]:
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)
        if precision in ["int8"]:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = EngineCalibrator(calib_cache)
            # if calib_cache is None or not os.path.exists(calib_cache):
            #     calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
            #     calib_dtype = trt.nptype(inputs[0].dtype)
            #     self.config.int8_calibrator.set_image_batcher(
            #         ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
            #                      exact_batches=True, config_file=config_file))

        engine_bytes = None
        try:
            engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        except AttributeError:
            engine = self.builder.build_engine(self.network, self.config)
            engine_bytes = engine.serialize()
            del engine
        assert engine_bytes
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)

    def load_engine(self, engine_path):
        # cuda.init()
        # device = cuda.Device(0)
        # self.cuda_ctx = device.make_context()
        self.dtype = np.int8
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.batch_size = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(None, "")   
        assert os.path.exists(engine_path)
        print("Reading engine from file {}".format(engine_path))          
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        return engine
onnx_model_path = '/opt/data/fcos_opset16.onnx'
onnx_simp_path = '/opt/data/fcos_opset16_simp_16.onnx'
engine_path = '/opt/data/fcos_fp16.engine'
graph = gs.import_onnx(onnx.load(onnx_model_path))
assert graph
log.info("ONNX graph loaded successfully")
graph.fold_constants()
for i in range(20):
    count_before = len(graph.nodes)
    graph.cleanup().toposort()
    try:
        for node in graph.nodes:
            for o in node.outputs:
                o.shape = None
        model = gs.export_onnx(graph)
        model = shape_inference.infer_shapes(model)
        graph = gs.import_onnx(model)
    except Exception as e:
        log.info("Shape inference could not be performed at this time:\n{}".format(e))
    try:
        graph.fold_constants(fold_shapes=True)
    except TypeError as e:
        log.error("This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                    "onnx_graphsurgeon module. Error:\n{}".format(e))
        raise

    count_after = len(graph.nodes)
    log.info("Reduced model nodes from {} to {} in iteration number {}".format(count_before, count_after, i))
    if count_before == count_after:
        # No new folding occurred in this iteration, so we can stop for now.
        log.info("Model has not been simplified any further. Saving model now")
        model = save(graph, onnx_simp_path)
        break
    if i==11:
        model = save(graph, onnx_simp_path)

builder = EngineBuilder(verbose=False, workspace=1)
builder.create_network(onnx_simp_path)
builder.create_engine(engine_path=engine_path,
                      precision='fp16',
                      config_file='/opt/git/detectron2-1/configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py'
                      )
engine = builder.load_engine(engine_path)
print("hold")