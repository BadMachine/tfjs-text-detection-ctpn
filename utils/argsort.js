import * as tf from '@tensorflow/tfjs-node-gpu';

export function argSort (tensor){
    const array = tensor.arraySync();
    const initial = Array.from(array);
    const sorted = array.sort((a, b)=>{return a-b});
    const args = sorted.map( item=>{ return initial.indexOf(item)})
    return tf.tensor1d(args);
}
