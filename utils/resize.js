import * as tf from '@tensorflow/tfjs-node-gpu';

export function resize_im(im, scale, max_scale=null) {
    let f = scale / Math.min(im.shape[0], im.shape[1]);
    if (max_scale != null && f * Math.max(im.shape[0], im.shape[1]) > max_scale) {
        f = max_scale / Math.max(im.shape[0], im.shape[1]);
    }
    const [newH, newW] = [im.shape[0] * f, im.shape[1] * f]
    return [tf.image.resizeBilinear(im, [newH, newW]), f]
}
