
<img  src="https://avatars2.githubusercontent.com/u/9568666?s=200&v=4"  alt="Lush Digital logo"  title="Lush Digital"  align="right"  height="96"  width="96"  />

# Lens Edge Library

This is a small wrapper library that can be used to run inference on a TFLite model in Python with the option of using a [Coral EdgeTPU](https://coral.withgoogle.com/products/accelerator/) for faster inference times.

The library uses the TFLite runtime package as a full TensorFlow installation includes a lot of functionality that is not required to run inference. Because of this, there is a requirement of Python version `3.5` or `3.7` on an `armhf` architecture.

## Setup

If a Coral EdgeTPU device is going to be utilised, firstly follow the installation process for the EdgeTPU runtime, which can be found on the [getting started](https://coral.withgoogle.com/docs/accelerator/get-started/) pages.

Once this is complete, install this package using the following

```bash
pip install https://github.com/LUSHDigital/lens-edge/archive/master.zip
```

## Usage

This is a super easy to use wrapper, so, the basic usage looks rather small. There are some more examples in the [examples](https://github.com/LUSHDigital/lens-edge/tree/master/examples) directory that can be looked at. But the basic use to run inference on a CPU is

``` python
import lens_edge

# Run inference on CPU
model = lens_edge.infer('TFLITE_MODEL_PATH', 'LABELS_PATH')
results = model.run('IMAGE_PATH')

print(results)
```

to run inference on a Coral EdgeTPU accelerator,  limiting to one return match with a minimum threshold of 50% use

```python
import lens_edge

# Run inference using Coral EdgeTPU adjusting count and threshold
model = lens_edge.infer('EDGETPU_TFLITE_MODEL_PATH', 'LABELS_PATH', 'libedgetpu.so.1')
results = model.run('IMAGE_PATH', count=1, threshold=0.5)

print(results)
```

## Credits

-  [Chris Hemmings](https://github.com/chrishemmings)
-  [All Contributors](https://github.com/LUSHDigital/lens-edge/contributors)

## License

The MIT License (MIT). Please see [License File](https://github.com/LUSHDigital/lens-edge/blob/master/LICENSE) for more information.
