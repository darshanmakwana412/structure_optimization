import jstorch from "./src/index.js";

class Graph2d {
    constructor(nodes, edges) {

        this.nodes = [];
        for(const node of nodes) {
            this.nodes.push(jstorch.tensor([node.x, node.y], true));
        }
        this.N = nodes.length;

        this.edges = [];
        for(const edge of edges) {
            var startidx = nodes.findIndex(node => node === edge.start);
            var endidx = nodes.findIndex(node => node === edge.end);
            this.edges.push([startidx, endidx]);
        }
        this.M = edges.length;

        var A = jstorch.zeros([2 * this.N, this.M]);
        for(let idx = 0; idx < this.M; idx++) {
            var n1 = this.nodes[this.edges[idx][0]]
            var n2 = this.nodes[this.edges[idx][1]]
            var [ux, uy] = this.proj(n1[0] - n2[0], n1[1] - n2[1]);
            console.log(A[n1 * 2])
            A[n1 * 2][idx] = ux
            A[n1 * 2 + 1][idx] = uy
            A[n2 * 2][idx] = -ux
            A[n2 * 2 + 1][idx] = -uy
        }
        // this.nodes = jstorch.tensor(nodes, {requires_grad: true});
        // this.edges = edges;
        // this.loads = [];
        // this.anchors = [];
        // this.mask = jstorch.ones_like(this.nodes);
    }

    proj(dx, dy) {
        let hypot = dx * dx + dy * dy;
        hypot = hypot.sqrt;
        return [dx / hypot, dy / hypot];
    }
}

export default Graph2d;