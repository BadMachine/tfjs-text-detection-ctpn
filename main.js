import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs';
import * as PImage from 'pureimage';
import { proposal_layer } from './rpn_msr/proposal_layer_tf.js';
import { resize_im } from "./utils/resize.js";
import { TextDetector } from './text_connector/detectors.js';
import { RGB2BGR } from './utils/RGB2BGR.js';
import { _get_blobs } from "./fast_rcnn/inference_blob.js";

export default class CTPN{
    constructor(config) {
        this.model = tf.loadGraphModel('https://cdn.jsdelivr.net/gh/BadMachine/tfjs-text-detection-ctpn/ctpn_web/model.json'); //tf.loadGraphModel('file://./ctpn_web/model.json');
        this.cfg = config;
    }

    async predict(image_path){
        const image = RGB2BGR(tf.node.decodeImage(fs.readFileSync(image_path)).cast('float32'));
        const [img, scale] = resize_im(image, 600, 1200);

        const [blobs, im_scales] = _get_blobs(img, null, this.cfg);
        if (this.cfg.HAS_RPN){
            const im_blob = blobs.data;
            blobs.im_info = tf.tensor( [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]]);
        }
        const model = await this.model;
        const raw = await model.executeAsync(img.expandDims());
        const [cls_prob, box_pred] = raw;
        let [scores, proposals, bbox_deltas] = await proposal_layer(this.cfg, cls_prob, box_pred, blobs.im_info,'TEST');
        const boxes = tf.div(proposals, im_scales[0]);
        const textDetector = new TextDetector(this.cfg);
        const _boxes = await textDetector.detect(boxes, scores.reshape([scores.shape[0],1]), img.shape.slice(0,2));
        return [_boxes, scale];
    }

    async draw(image_name, writeTo, _boxes, scale, color){
        const image = await PImage.decodeJPEGFromStream(fs.createReadStream(`./${image_name}`));

        const boxes = _boxes.arraySync();
        for(let box of boxes){
            if (tf.norm(box[0] - box[1]) < 5 || tf.norm(box[3] - box[0]) < 5) continue;

            const ctx = image.getContext('2d');
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.moveTo(box[0]/ scale, box[1]/ scale);
            ctx.lineTo(box[2] / scale, box[3] / scale);

            ctx.lineTo(box[0] / scale, box[1] / scale);
            ctx.lineTo(box[4] / scale, box[5] / scale);

            ctx.lineTo(box[6] / scale, box[7] / scale);
            ctx.lineTo(box[2] / scale, box[3] / scale);

            ctx.stroke();
            ctx.closePath();

        }

        PImage.encodeJPEGToStream(image,fs.createWriteStream(writeTo), 50).then(() => {
            console.log("done writing");
        });

    }
}
