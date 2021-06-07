import * as tf from '@tensorflow/tfjs-node-gpu';

export class Graph{
    constructor(graph) {
        this.graph = graph;
    }

    async sub_graphs_connected(){
        const sub_graphs = [];
        for (let index = 0; index < this.graph.shape[0]; index++){

            const firstCondition = this.graph.slice([0,index], [this.graph.shape[0],1]).reshape([this.graph.shape[0]]).any().logicalNot();
            const secondCondition = this.graph.slice([index,0], [1 ,this.graph.shape[0]]).reshape([this.graph.shape[0]]).any();
            const condition = tf.logicalAnd(firstCondition, secondCondition).arraySync();

            if(condition) {
                let v = index;
                sub_graphs.push([v]);

                while (tf.gather(this.graph, v).any().arraySync()){
                    v = await tf.whereAsync(tf.gather(this.graph, v))
                    v = v.arraySync()[0][0];
                    sub_graphs[sub_graphs.length-1].push(v);
                }
            }
        }
        return sub_graphs;
    }
}
