import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import collections
import operator
import lens_edge.classify as classify
import time

class infer:

    labels = None
    interpreter = None
    class_collection = collections.namedtuple('Class', ['id', 'score'])
    last_inference_time = None

    def __init__(self, model_path, labels_path, edge_tpu_lib = None):
        """Construct the object, loading the model and label"""
        self.load_labels(labels_path)
        self.construct_interpreter(model_path, edge_tpu_lib)

    def run(self, image, threshold = 0, count = 5, rgb2bgr = True, img_std = 255.0):
        """Infer an image from the passed filename, or, cv2 loaded image"""
        # Check we have loaded some labels
        if (self.labels is None):
            raise Exception('No labels set')

        # Check we have instantiated a interpreter with a model
        if (self.interpreter is None):
            raise Exception('No model set')

        input_size = classify.input_size(self.interpreter)

        # Load image, if image has been passed as a string
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Resize image if required
        if tuple(image.shape[1::-1]) != input_size:
            image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
        
        # Change from RGB to BGR if needed, depending on how model was trained
        if rgb2bgr == True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Noramlize image
        image = np.expand_dims(image.astype(np.float32), axis=0) / img_std

        start = time.monotonic()
        classify.set_input(self.interpreter, image)
        self.interpreter.invoke()
        classes = classify.get_output(self.interpreter, count, threshold)
        self.last_inference_time = (time.monotonic() - start) * 1000

        results = {}
        for klass in classes:
            results[self.labels.get(klass.id, klass.id)] = klass.score

        return results

    def load_labels(self, path, encoding='utf-8'):
        """Loads labels from the file passed in, one lable per line"""
        self.labels = None

        with open(path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                return {}
                
            if lines[0].split(' ', maxsplit=1)[0].isdigit():
                pairs = [line.split(' ', maxsplit=1) for line in lines]
                self.labels = {int(index): label.strip() for index, label in pairs}
            else:
                self.labels = {index: line.strip() for index, line in enumerate(lines)}
                
    def construct_interpreter(self, model_file, edge_tpu_lib):
        """Constructs a tflite Interpreter"""
        self.interpreter = None
        model_file, *device = model_file.split('@')

        delegates = []
        if edge_tpu_lib != None:
            delegates = [tflite.load_delegate(edge_tpu_lib, {'device': device[0]} if device else {})]

        interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=delegates
        )

        interpreter.allocate_tensors()
        self.interpreter = interpreter
