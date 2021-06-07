import * as tf from '@tensorflow/tfjs-node-gpu';


function scale_anchor(anchor, h, w) {
    const x_ctr = (anchor[0] + anchor[2]) * 0.5;
    const y_ctr = (anchor[1] + anchor[3]) * 0.5;

    const scaled_anchor = Array.from(anchor);
    scaled_anchor[0] = ~~(x_ctr - w / 2);
    // xmin
    scaled_anchor[2] = ~~(x_ctr + w / 2);
    // xmax
    scaled_anchor[1] = ~~(y_ctr - h / 2);
    // ymin
    scaled_anchor[3] = ~~(y_ctr + h / 2);
    // ymax
    return scaled_anchor;
}

function generate_basic_anchors(sizes, base_size= 16) {
    const base_anchor = [0, 0, base_size - 1, base_size - 1];
    const anchors = tf.buffer([sizes.length, 4], 'int32');
    let index = 0;
    for (let item of sizes) {
        const [x, y, z ,w] = scale_anchor(base_anchor, item[0], item[1]);
        anchors.set(x, index, 0);
        anchors.set(y, index, 1);
        anchors.set(z, index, 2);
        anchors.set(w, index, 3);
        index += 1;
    }

    return anchors.toTensor();
}
export function generate_anchors(base_size= 16, ratios= [0.5, 1, 2], scales= [3**2,4**2,5**2,6**2]) {
    const heights = tf.tensor1d([11, 16, 23, 33, 48, 68, 97, 139, 198, 283]);
    const widths = tf.tensor1d([16]);
    const sizes = tf.stack([heights, widths.tile(heights.shape)], 1);

    return generate_basic_anchors(sizes.arraySync());
}
