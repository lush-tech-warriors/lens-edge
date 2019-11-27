import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lens_edge",
    version="0.0.1",
    author="Chris Hemmings",
    author_email="chris.hemmings@lush.co.uk",
    description="A small library to use the Coral EdgeTPU Device on a Raspberry Pi4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lushdigital/lens_edge",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy >= 1.16.2',
        'opencv-python >= 3.4.3.18',
        'tflite-runtime@https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl ; platform_machine=="armv7l" and python_version=="3.5"',
        'tflite-runtime@https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl ; platform_machine=="armv7l" and python_version=="3.7"'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, !=3.6.*'
)
