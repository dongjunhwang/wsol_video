![wsol_bird](https://user-images.githubusercontent.com/51294226/154967766-85e020b5-7301-44cf-aa22-f180e0204325.gif)
## Weakly Supervised Object Localization in Video Frames
Use 7 Weakly Supervised Object Localization (WSOL) Methods to localize the object in video frames

**7 Papers are used**  
- Learning Deep Features for Discriminative Localization (CAM)
- Attention Based Dropout Layer for WSOL (ADL)
- Adversarial Complementary Learning for WSOL (ACoL)
- Hide and Seek (HaS)
- CutMix
- Self-produced Guidance for WSOL (SPG)
- Normalization Matters in WSOL (IVR)

**Reference Code**  
https://github.com/clovaai/wsolevaluation  

If you want to execute and debug this code, please note `make_video.py` is the execution files.

Please configure your command line like below.

```python
python make_video.py --dataset_name ILSVRC --wsol_method cam
```

Want to use this code your own, analyzed the `config.py` for add the options.


