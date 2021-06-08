import * as tf from '@tensorflow/tfjs-node-gpu';
import {ravel} from "../utils/ravel.js";
import {argSort} from "../utils/argsort.js";
export async function nms(config){
    return (config.method=='TF')? tf_nms(config.dets, config.scores, config.thresh) : authNMS(config.dets, config.scores, config.thresh);
}

function tf_nms(dets, scores, thresh){
    return tf.image.nonMaxSuppression(dets, ravel(scores), 2000, 0.2,thresh);
}

async function authNMS(dets, scores, thresh){

    const x1 = dets.slice([0,0], [dets.shape[0],1]).squeeze(); //x1 = dets[:, 0]
    const y1 = dets.slice([0,1], [dets.shape[0],1]).squeeze(); //y1 = dets[:, 1]
    const x2 = dets.slice([0,2], [dets.shape[0],1]).squeeze(); //x2 = dets[:, 2]
    const y2 = dets.slice([0,3], [dets.shape[0],1]).squeeze(); //y2 = dets[:, 3]
    //const scores = dets.slice([0,4], [dets.shape[0],1]).squeeze(); //y2 = dets[:, 3]
    let areas = tf.mul( x2.sub(x1).add(1), y2.sub(y1).add(1)) ;//areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    let order = argSort(scores).reverse();//order = scores.argsort()[::-1]

    order = ravel(order);
    let keep = tf.tensor1d([]);
    while(order.shape[0] > 0){
        const i = order.slice(0,1).cast('int32');
        keep = keep.concat(i.cast('float32'));
        const xx1 = tf.maximum(x1.gather(i), x1.gather(order.slice(1,-1).cast('int32') )); //xx1 = np.maximum(x1[i], x1[order[1:]])
        const yy1 = tf.maximum(y1.gather(i), y1.gather(order.slice(1,-1).cast('int32') )); //yy1 = np.maximum(y1[i], y1[order[1:]])
        const xx2 = tf.minimum(x2.gather(i), x2.gather(order.slice(1,-1).cast('int32') )); //xx2 = np.minimum(x2[i], x2[order[1:]])
        const yy2 = tf.minimum(y2.gather(i), y2.gather(order.slice(1,-1).cast('int32') )); //yy2 = np.minimum(y2[i], y2[order[1:]])
        const w = tf.maximum(0.0, xx2.sub(xx1).add(1) ); //w = np.maximum(0.0, xx2 - xx1 + 1)
        const h = tf.maximum(0.0, yy2.sub(yy1).add(1) ); //h = np.maximum(0.0, yy2 - yy1 + 1)
        const inter = w.mul(h);
        const ovr = tf.div(inter, ( (areas.gather(i).add(areas.gather(order.slice(1,-1).cast('int32') ))).sub(inter) ));
        let inds = await tf.whereAsync( ovr.lessEqual(thresh) ); // here is a bottleneck
        inds = ravel(inds);
        order = order.gather(inds.add(1).cast('int32') );

    }

    return keep;
}

