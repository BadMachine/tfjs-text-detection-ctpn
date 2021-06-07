import {TextLineCfg} from './TextLineCfg.js';
import {Graph} from './other.js';
import * as tf from '@tensorflow/tfjs-node-gpu';
import {argmax} from '../utils/argmax.js';
import {ravel} from "../utils/ravel.js";

export class TextProposalGraphBuilder{
    constructor() {
    }

    get_successions(index){

        const box = this.text_proposals[index];
        const results=[];
        for(let left = Math.round(box[0])+1; left < Math.min(Math.round(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP+1, this.im_size[1]); left++){ // for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
           const adj_box_indices = this.boxes_table[left]; // adj_box_indices=self.boxes_table[left]
            for (let adj_box_index of adj_box_indices){
                if (this.meet_v_iou(adj_box_index, index)) results.push(adj_box_index);
            }

            if (results.length!==0) return results;
        }
        return results;
    }

    get_precursors(index) {
       const box = this.text_proposals[index];
       const results = [];

        for(let left = Math.round(box[0])-1; left > Math.max(Math.round(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) -1; left--){
            const adj_box_indices = this.boxes_table[left];
            for (let adj_box_index of adj_box_indices){
                if (this.meet_v_iou(adj_box_index, index)) results.push(adj_box_index);
            }
            if (results.length!==0) return results;
        }
        return results;
    }

    meet_v_iou(index1, index2){
        const overlaps_v = (index1, index2) =>{
            const h1 = this.heights[index1];
            const h2 = this.heights[index2];
            const y0 = Math.max(this.text_proposals[index2][1], this.text_proposals[index1][1]);
            const y1 = Math.min(this.text_proposals[index2][3], this.text_proposals[index1][3]);
            return Math.max(0, y1-y0+1)/Math.min(h1, h2);
        }

        const size_similarity = (index1, index2)=>{
           const h1 = this.heights[index1]
           const h2 = this.heights[index2]
            return Math.min(h1, h2) / Math.max(h1, h2);
        }

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS && size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM;
    }

    is_succession_node(index, succession_index) {
        const precursors = this.get_precursors(succession_index);
        return this.scores[index] >= Math.max(this.scores[precursors]) ? true : false
    }
    build_graph(text_proposals, scores, im_size){
        this.text_proposals = text_proposals.arraySync();
        this.scores = ravel(scores).arraySync();
        this.im_size = im_size;
        const h1 = text_proposals.slice([0,3], [text_proposals.shape[0],1]).reshape([text_proposals.shape[0]]);
        const h2 = text_proposals.slice([0,1], [text_proposals.shape[0],1]).reshape([text_proposals.shape[0]]);
        this.heights = tf.add(tf.sub(h1,h2), 1).arraySync();
        const boxes_table =  Array.from(Array(im_size[1]), () => []);
        this.text_proposals.forEach((item, index)=>{ // here probably undry
             boxes_table[Math.round(item[0])].push(index);
         })
        this.boxes_table = boxes_table;
        let graph = tf.zeros([text_proposals.shape[0], text_proposals.shape[0]], 'bool').arraySync();
        for(let index = 0; index < this.text_proposals.length; index++){
            let successions = this.get_successions(index);
            if (successions.length === 0) continue;
            if (successions.length > 1) successions = [successions[successions.length-1]]; //weak fix

            const succession_index = successions[argmax(this.scores[successions])];//succession_index=successions[np.argmax(scores[successions])]
            if (this.is_succession_node(index, succession_index)) {
                graph[index][succession_index] = true;
            }

        }
        graph = tf.tensor(graph).cast('bool');
        return new Graph(graph);
    }
}
