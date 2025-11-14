/*
Simple 2D graph viewer using three.js (rendered in 3D scene but constrained to Z=0)
Loads s6_graph.json (nodes, edges) and s6_stats.json (edge stats) and renders nodes colored by category.
*/

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let nodes = []; // { id, title, category, x, y, mesh }
let edges = []; // { source, target, similarity, status, line }
let nodeIndex = new Map();
let categoryColors = new Map();
let defaultColors = [
  '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#22c55e', '#eab308'
];
let colorCursor = 0;

const canvas = document.getElementById('canvas');
const legendEl = document.getElementById('legend');
const statsEl = document.getElementById('stats');
const graphFileEl = document.getElementById('graphFile');
const statsFileEl = document.getElementById('statsFile');
const loadDefaultBtn = document.getElementById('loadDefault');
const resetViewBtn = document.getElementById('resetView');

init();

function init() {
  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-50, 50, 30, -30, 0.1, 1000);
  camera.position.set(0, 0, 100);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  onResize();
  window.addEventListener('resize', onResize);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.zoomSpeed = 1.2;
  controls.enablePan = true;

  animate();

  loadDefaultBtn.addEventListener('click', () => loadFromPaths('../test/output/s6_graph.json', '../test/output/s6_stats.json'));
  resetViewBtn.addEventListener('click', resetView);
  graphFileEl.addEventListener('change', handleGraphFile);
  statsFileEl.addEventListener('change', handleStatsFile);
}

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.left = -w / 30;
  camera.right = w / 30;
  camera.top = h / 30;
  camera.bottom = -h / 30;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

async function loadFromPaths(graphPath, statsPath) {
  try {
    const [graphRes, statsRes] = await Promise.all([
      fetch(graphPath),
      fetch(statsPath).catch(() => null),
    ]);
    if (!graphRes.ok) throw new Error('Failed to load graph JSON');
    const graph = await graphRes.json();
    let stats = null;
    if (statsRes && statsRes.ok) stats = await statsRes.json();
    applyGraph(graph, stats);
  } catch (e) {
    alert('Error loading files: ' + e.message);
  }
}

function handleGraphFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const graph = JSON.parse(ev.target.result);
    applyGraph(graph, null);
  };
  reader.readAsText(file);
}

function handleStatsFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const stats = JSON.parse(ev.target.result);
    updateStats(stats);
  };
  reader.readAsText(file);
}

function applyGraph(graph, stats) {
  clearScene();
  const { nodes: nodeList, edges: edgeList } = graph;

  // Build node index and category colors
  nodeIndex.clear();
  categoryColors.clear();
  colorCursor = 0;
  nodes = nodeList.map((n, i) => ({ ...n, x: 0, y: 0, mesh: null }));
  nodes.forEach((n, idx) => nodeIndex.set(n.id, idx));

  // Simple circular layout
  const R = Math.max(20, nodes.length * 0.8);
  nodes.forEach((n, i) => {
    const angle = (i / nodes.length) * Math.PI * 2;
    n.x = Math.cos(angle) * R;
    n.y = Math.sin(angle) * R;
  });

  // Create node meshes
  nodes.forEach(n => {
    const color = colorForCategory(n.category);
    const geom = new THREE.CircleGeometry(0.8, 24);
    const mat = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(n.x, n.y, 0);
    mesh.userData = { title: n.title, category: n.category, id: n.id };
    scene.add(mesh);
    n.mesh = mesh;
  });

  // Create edges
  edges = edgeList.map(e => ({ ...e, line: null }));
  edges.forEach(e => {
    const a = nodes[nodeIndex.get(e.source)];
    const b = nodes[nodeIndex.get(e.target)];
    if (!a || !b) return;
    const color = e.status === 'hard' ? 0x111827 : 0x9ca3af; // dark for hard, grey for pending
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 });
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(a.x, a.y, 0),
      new THREE.Vector3(b.x, b.y, 0)
    ]);
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    e.line = line;
  });

  updateLegend(nodes);
  updateStats(stats);
}

function updateLegend(nodes) {
  const el = legendEl;
  el.innerHTML = '';
  const cats = Array.from(new Set(nodes.map(n => n.category).filter(Boolean)));
  cats.forEach(c => {
    const color = colorForCategory(c);
    const item = document.createElement('div');
    item.className = 'legend-item';
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = color;
    const label = document.createElement('div');
    label.textContent = c;
    item.appendChild(sw);
    item.appendChild(label);
    el.appendChild(item);
  });
}

function updateStats(stats) {
  const el = statsEl;
  if (!stats) {
    el.textContent = 'Stats not loaded';
    return;
  }
  el.innerHTML = '';
  const lines = [
    `BERTopic time: ${stats.bertopic_seconds ?? '-'}s`,
    `Total edges: ${stats.total_edges ?? '-'}`,
    `Within-category edges: ${stats.within_category_edges ?? '-'}`,
    `Cross-category edges: ${stats.cross_category_edges ?? '-'}`,
  ];
  el.innerHTML = lines.map(l => `<div>${l}</div>`).join('');
}

function colorForCategory(cat) {
  if (!cat) return '#6b7280'; // gray for unknown
  if (!categoryColors.has(cat)) {
    const col = defaultColors[colorCursor % defaultColors.length];
    categoryColors.set(cat, col);
    colorCursor++;
  }
  return categoryColors.get(cat);
}

function resetView() {
  camera.position.set(0, 0, 100);
  controls.target.set(0, 0, 0);
  controls.update();
}

function clearScene() {
  while (scene.children.length > 0) {
    const obj = scene.children.pop();
    obj.geometry && obj.geometry.dispose && obj.geometry.dispose();
    obj.material && obj.material.dispose && obj.material.dispose();
  }
}
