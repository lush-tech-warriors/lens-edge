import lens_edge

# Run inference on CPU
model = lens_edge.infer('TFLITE_MODEL_PATH', 'LABELS_PATH')
results = model.run('IMAGE_PATH')
print(results)

# Run inference using Coral EdgeTPU adjusting count and threshold
model = lens_edge.infer('EDGETPU_TFLITE_MODEL_PATH', 'LABELS_PATH', 'libedgetpu.so.1')
results = model.run('IMAGE_PATH', count=1, threshold=0.5)
print(results)

# Run inference on an CV2 loaded image
cv2_image = cv2.imread('IMAGE_PATH')
results = model.run(cv2_image, count=1, threshold=0.5)
print(results)
