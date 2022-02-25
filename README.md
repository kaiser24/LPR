## LIcense Plate Recognition
### License Plate Recognition system using a license plate detector on tensorflow and OCR using YoloV3

#### To install run
```/bin/bash install.sh```

#### To use it activate the venv and a couple of Env variables
```source lpr_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$PATH"
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda/'```

#### Then run using one of the following commands:
##### To run from console, using processing the whole image
```python3 LPR.py -i '/route/to/video' -s```
##### To run and drawing a sub area where to focus the analysis
```python3 LPR.py -i '/route/to/video' -dz -s```
##### To run from console and passing the polygon ( where to focus the analysis) coordinates directly in json format
```python3 LPR.py -i '/route/to/video' -iz '[{"x": 150, "y": 217}, {"x": 97, "y": 308}, {"x": 561, "y": 299}, {"x": 551, "y": 227}, {"x": 150, "y": 217}]' -s```

##### Input parameters details
##### -i --input: Media source. Video, Stream or Image")
##### -iz --zone: Points that determine the zone where to reduce the detection. as a json. Exmaple: 
#####                                                      Example: '[{"x": 10, "y": 10}, {"x": 100, "y": 120},{...}]'
##### -dz --draw_zone: If selected allows to draw the input zone manually. this option ignores the flag -iz
##### -s --show_img: Shows image output
##### -d --debug: Sets Logger level to debug

#### TODO: Explain how to import the module on a project. how does it work. how to run giving input parameter or to run drawing an image. 

