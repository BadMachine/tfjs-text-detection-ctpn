import * as tf from '@tensorflow/tfjs-node-gpu';
import {argmax} from "../utils/argmax.js";

export function _get_blobs(img, rois, cfg){
    const blobs = {
        data : null,
        rois : null,
        im_info: null
    };
    let im_scale_factors;
    [blobs.data, im_scale_factors] = _get_image_blob(img, cfg);
    return [blobs, im_scale_factors];
}


function _get_image_blob(im, cfg){

    let im_orig = im.cast('float32');
    im_orig = im_orig.sub(cfg.PIXEL_MEANS);

    const im_shape = im_orig.shape;
    const [w, h] = im_shape.slice(0,2);
    const im_size_min = Math.min(w, h);
    const im_size_max = Math.max(w,h);
    const processed_ims = [];
    const im_scale_factors = [];

    for (let target_size of cfg.SCALES){
        let im_scale = target_size / im_size_min;
        // Prevent the biggest axis from being more than MAX_SIZE
        if (Math.round(im_scale * im_size_max) > cfg.MAX_SIZE){
            im_scale = cfg.MAX_SIZE / im_size_max;
        }
        im = tf.image.resizeBilinear(im_orig, [w * im_scale, h * im_scale])
        im_scale_factors.push(im_scale);
        processed_ims.push(im);
    }
    // Create a blob to hold the input images
    const blob = im_list_to_blob(processed_ims);
    return [blob, im_scale_factors];
}

function im_list_to_blob(ims){

    // Convert a list of images into a network input.
    //
    // Assumes images are already prepared (means subtracted, BGR order, ...).
    const max_shape = ims[argmax(ims.map(im=>im.shape[0] * im.shape[1] * im.shape[2]))].shape;

    const num_images = ims.length;
    let blob = tf.zeros([num_images, max_shape[0], max_shape[1], 3], 'float32').arraySync();

    for (let i in num_images){
        const im = ims[i];
        blob[i] = im.arraySync();
    }

    blob = tf.tensor(blob);

    return blob;
}
