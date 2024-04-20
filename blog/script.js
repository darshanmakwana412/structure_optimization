import jstorch from "./src/index.js";
import Graph2D from "./graph2d.js";

const canvas = document.getElementById('networkCanvas');
const ctx = canvas.getContext('2d');
const canvasContainer = document.getElementById('canvas-container'); // Get container

const nodes = [];
const edges = [];
const forces = [];
let isConnecting = false;
let isAddingForces = false;
let startConnectingNode = null; // Stores the starting node for connection
let startForceNode = null; // Stores the starting node for force
let tempForceEnd = null; // Temporary end coordinate for force while adding

function resizeCanvas() {
    canvas.width = canvasContainer.offsetWidth;
    canvas.height = canvasContainer.offsetHeight;
}

// Initial resize
resizeCanvas();

// Resize on window resize
window.addEventListener('resize', resizeCanvas); 

function drawNode(x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawAnchor(x, y, size, color) {
    ctx.beginPath();
    ctx.rect(x - size / 2, y - size / 2, size, size);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 8;
    ctx.strokeStyle = 'red';
    ctx.stroke();
}

function drawArrow(x1, y1, x2, y2, color) {
    const arrowLength = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    const headLength = arrowLength * 0.5; // 30% of the length of the arrow
    const lineWidth = arrowLength * 0.2; // Increased width of the arrow
    const dx = x2 - x1;
    const dy = y2 - y1;
    const angle = Math.atan2(dy, dx);

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLength * Math.cos(angle - Math.PI / 6), y2 - headLength * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headLength * Math.cos(angle + Math.PI / 6), y2 - headLength * Math.sin(angle + Math.PI / 6));
    ctx.lineTo(x2, y2);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2 - 0.4 * arrowLength * Math.cos(angle), y2 - 0.4 * arrowLength * Math.sin(angle));
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function edgeExists(node1, node2) {
    return edges.some(edge => 
        (edge.start === node1 && edge.end === node2) || 
        (edge.start === node2 && edge.end === node1)
    );
}

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw edges
    for (const edge of edges) {
        drawLine(edge.start.x, edge.start.y, edge.end.x, edge.end.y);
    }

    // Draw forces
    for (const force of forces) {
        drawArrow(force.start.x, force.start.y, force.end.x, force.end.y, 'green');
    }
    // Draw temporary force
    if (tempForceEnd) {
        drawArrow(startForceNode.x, startForceNode.y, tempForceEnd.x, tempForceEnd.y, 'rgba(0, 255, 0, 0.8)');
    }

    // Draw nodes
    for (const node of nodes) {
        if(!node.fixed) {
            drawNode(node.x, node.y, 5, 'blue');
        } else {
            drawAnchor(node.x, node.y, 20, 'green');
        }
    }

    const g = new Graph2D(nodes, edges);

}

let canAddNodes = false; 
let canAddAnchors = false;

canvas.addEventListener('mousedown', (e) => {
    const clickX = e.offsetX;
    const clickY = e.offsetY;
    const clickedNode = nodes.find(node => 
        Math.sqrt((node.x - clickX) ** 2 + (node.y - clickY) ** 2) < 10
    );

    if (canAddNodes) {
        if (!clickedNode) {
            nodes.push({ x: clickX, y: clickY , fixed: false});
            render();
        }
    } else if (canAddAnchors) {
        clickedNode.fixed = !clickedNode.fixed;
        render();
    } else if (isConnecting) {
        if (clickedNode) {
            if (startConnectingNode) {
                if (!edgeExists(startConnectingNode, clickedNode)) {
                    edges.push({ start: startConnectingNode, end: clickedNode });
                    render();
                }
                startConnectingNode = null; // Reset after connecting
            } else {
                startConnectingNode = clickedNode;
            }
        }
    } else if (isAddingForces) {
        if (startForceNode) {
            forces.push({ start: startForceNode, end: {x: clickX, y: clickY} });
            tempForceEnd = null; // Clear temporary force end
            render();
            startForceNode = null; // Reset after adding force
        } else if (clickedNode) {
            startForceNode = clickedNode;
        }
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (isAddingForces && startForceNode) {
        tempForceEnd = { x: e.offsetX, y: e.offsetY };
        render();
    }
});

document.getElementById('addNode').addEventListener('click', () => {
    canAddNodes = !canAddNodes;
    if (canAddNodes) {
        isConnecting = false; // Disable connecting when adding nodes
        canAddAnchors = false; // Disable adding anchors
        isAddingForces = false; // Disable adding forces
    }
});

document.getElementById('addAnchor').addEventListener('click', () => {
    canAddAnchors = !canAddAnchors;
    if (canAddAnchors) {
        canAddNodes = false; // Disable node adding
        isConnecting = false; // Disable connecting
        isAddingForces = false; // Disable adding forces
    }
});

document.getElementById('connectNodes').addEventListener('click', () => {
    isConnecting = !isConnecting;
    canAddNodes = false; // Disable node adding while connecting 
    canAddAnchors = false; // Disable adding anchors
    isAddingForces = false; // Disable adding forces
    startConnectingNode = null; 
});

document.getElementById('addForce').addEventListener('click', () => {
    isAddingForces = !isAddingForces;
    canAddNodes = false; // Disable node adding
    canAddAnchors = false; // Disable adding anchors
    isConnecting = false; // Disable connecting
    startForceNode = null;
    tempForceEnd = null; // Reset temporary force end
});

setInterval(render, 30);