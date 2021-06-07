import * as tf from '@tensorflow/tfjs-node-gpu';

export function ravel(tensor){
    return tf.tensor1d( tf.util.flatten(tensor.arraySync()));
}
