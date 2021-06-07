import * as tf from '@tensorflow/tfjs-node-gpu';

export function bbox_transform_inv(boxes, deltas){

    boxes = boxes.cast(deltas.dtype);

    const w1 = boxes.slice([0,2], [boxes.shape[0],1]).reshape([boxes.shape[0]]);
    const w2 = boxes.slice([0,0], [boxes.shape[0],1]).reshape([boxes.shape[0]]);
    const h1 = boxes.slice([0,3], [boxes.shape[0],1]).reshape([boxes.shape[0]]);
    const h2 = boxes.slice([0,1], [boxes.shape[0],1]).reshape([boxes.shape[0]]);

    const widths = tf.add(tf.sub(w1, w2), 1);
    const heights = tf.add(tf.sub(h1,h2), 1);

    const ctr_x = w2.add(tf.mul(0.5,widths));
    const ctr_y = h2.add(tf.mul(0.5,heights));

    //const dx = deltas.slice([0,0], [deltas.shape[0],1]);
    const dy = deltas.slice([0,1], [deltas.shape[0],1]);
    //const dw = deltas.slice([0,2], [deltas.shape[0],1]);
    const dh = deltas.slice([0,3], [deltas.shape[0],1]);


    const pred_ctr_x = ctr_x.reshape([ctr_x.shape[0],1]);
    const pred_ctr_y = tf.add( tf.mul( dy,   heights.reshape([heights.shape[0],1]) ),   ctr_y.reshape([ctr_y.shape[0],1]) );//pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    const pred_w = widths.reshape([widths.shape[0],1]);  //pred_w = widths[:, np.newaxis]
    const pred_h = tf.mul( tf.exp(dh), heights.reshape([heights.shape[0],1]) ); //pred_h = np.exp(dh) * heights[:, np.newaxis]

    const x1 = pred_ctr_x.sub(tf.mul(0.5, pred_w) );
    const y1 = pred_ctr_y.sub(tf.mul(0.5, pred_h) );
    const x2 = pred_ctr_x.add(tf.mul(0.5, pred_w) );
    const y2 = pred_ctr_y.add(tf.mul(0.5, pred_h) );

    return tf.stack( [x1, y1, x2, y2], -2 ).reshape(deltas.shape);
}

export function clip_boxes(boxes, im_shape){
// Clip boxes to image boundaries.
// // x1 >= 0
    const b1 = tf.maximum( tf.minimum (boxes.slice([0,0], [boxes.shape[0],1]), im_shape[1] -1), 0);
// // y1 >= 0
    const b2 = tf.maximum( tf.minimum (boxes.slice([0,1], [boxes.shape[0],1]), im_shape[0] -1), 0);
// // x2 < im_shape[1]
    const b3 = tf.maximum( tf.minimum (boxes.slice([0,2], [boxes.shape[0],1]), im_shape[1] -1), 0);
// // y2 < im_shape[0]
    const b4 = tf.maximum( tf.minimum (boxes.slice([0,3], [boxes.shape[0],1]), im_shape[0] -1), 0);

    return tf.stack( [b1, b2, b3, b4], -2 ).reshape(boxes.shape);

}
