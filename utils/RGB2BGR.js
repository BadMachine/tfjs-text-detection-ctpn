import * as tf from '@tensorflow/tfjs-node-gpu';

export function RGB2BGR(image){
    return tf.reverse(image, -1);
}
