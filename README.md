# tfjs-text-detection-ctpn
TensorflowJS text detection implementation (inference), mainly based on ctpn model in tensorflow, id card detect, connectionist text proposal network
The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). If you got any questions, check the issue first, if the problem persists, open a new issue.

## Setup
```
npm i tfjs-text-detection-ctpn
```
## Config object, that have to be passed to constructor:
```js
   {
     NMS_FUNCTION: 'AUTH' | 'TF', specifies which non maximum suppression function we gonna use. TF faster, AUTH more accurate
     ANCHOR_SCALES: [16],
     PIXEL_MEANS: tf.tensor([[[102.9801, 115.9465, 122.7717]]]),
     SCALES: [600,] , // model input layer preferable size
     MAX_SIZE:  1000, 
     HAS_RPN: true,
     DETECT_MODE: 'O' | 'H', // 'O' - Oriented mode (boxes will be rotated by angle), now supports only oriented one ( H - horisontal)
     pre_nms_topN: 12000,
     post_nms_topN: 2000,
     nms_thresh:0.7, // threshold for nms function
     min_size: 8,
   }
```

# Example

```js
import * as tf from '@tensorflow/tfjs-node-gpu';
import CTPN from 'tfjs-text-detection-ctpn';

(async ()=> {
    const cfg = {
        NMS_FUNCTION: 'AUTH',
        ANCHOR_SCALES: [16],
        PIXEL_MEANS: tf.tensor([[[102.9801, 115.9465, 122.7717]]]),
        SCALES: [600,] ,
        MAX_SIZE:  1000,
        HAS_RPN: true,
        DETECT_MODE: 'O',
        pre_nms_topN: 12000,
        post_nms_topN: 2000,
        nms_thresh:0.7,
        min_size: 8,
    };
    const ctpn = new CTPN(cfg);
    const image = './test/007.jpg';
    const predicted = await ctpn.predict(image);
    console.log(predicted);
    ctpn.draw(image,'res.jpg',...predicted, 'black')
})();

```



# Some results

<img src="/test/007.jpg" width=320 height=240 /><img src="/data/res_oriented/007.jpg" width=320 height=240 />
<img src="/data/res/008.jpg" width=320 height=480 /><img src="/data/res_oriented/008.jpg" width=320 height=480 />
***
