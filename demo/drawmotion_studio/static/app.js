import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const trajectoryCanvas = document.getElementById("trajectoryCanvas");
const poseCanvas = document.getElementById("poseCanvas");
const trajectoryCtx = trajectoryCanvas.getContext("2d");
const poseCtx = poseCanvas.getContext("2d");
const generateBtn = document.getElementById("generateBtn");
const textPrompt = document.getElementById("textPrompt");
const statusText = document.getElementById("statusText");
const frameText = document.getElementById("frameText");
const poseTime = document.getElementById("poseTime");
const keyframeList = document.getElementById("keyframeList");
const progressSlider = document.getElementById("progressSlider");
const progressMarkers = document.getElementById("progressMarkers");
const trajectoryScaleInput = document.getElementById("trajectoryScale");
const trajectoryScaleValue = document.getElementById("trajectoryScaleValue");
const ifgRepeatInput = document.getElementById("ifgRepeat");
const ifgScaleInput = document.getElementById("ifgScale");
const trajectoryAlphaInput = document.getElementById("trajectoryAlpha");
const trajectoryFramesInput = document.getElementById("trajectoryFrames");
const stickmanPanel = document.getElementById("stickmanPanel");
const generationProgress = document.getElementById("generationProgress");
const generationProgressBar = document.getElementById("generationProgressBar");
const generationProgressText = document.getElementById("generationProgressText");

let trajectory = [];
let trajectoryScale = Number(trajectoryScaleInput.value);
let drawingTrajectory = false;
let poseLines = [];
let currentPoseLine = [];
let drawingPose = false;
let keyframes = [];
let result = null;
let playing = true;
let frame = 0;
let lastFrameTime = 0;
let sceneTransform = { cx: 0, cz: 0, scale: 1 };
let cameraTarget = new THREE.Vector3(0, 1, 0);
let draggingProgress = false;
let generationProgressTimer = 0;
let generationProgressHideTimer = 0;
const playbackFps = 30;
const playbackFrameMs = 1000 / playbackFps;
const trajectoryPixelsPerMeter = 100;
const carLengthMeters = 4.8;
const carReference = new Image();
carReference.src = "/static/car_top.svg?v=20260512alpha";
carReference.addEventListener("load", drawTrajectoryCanvas);

const kitPairs = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10]];
const t2mPairs = [[0, 2], [2, 5], [5, 8], [8, 11], [0, 1], [1, 4], [4, 7], [7, 10], [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [9, 14], [14, 17], [17, 19], [19, 21], [9, 13], [13, 16], [16, 18], [18, 20]];

function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return { x: (event.clientX - rect.left) * sx, y: (event.clientY - rect.top) * sy };
}

function scaledTrajectoryPoints() {
  if (trajectory.length === 0) return [];
  const origin = trajectory[0];
  return trajectory.map((point) => ({
    x: origin.x + (point.x - origin.x) * trajectoryScale,
    y: origin.y + (point.y - origin.y) * trajectoryScale
  }));
}

function removeDuplicatePoints(points, eps = 0.25) {
  if (points.length === 0) return [];
  const filtered = [points[0]];
  for (let index = 1; index < points.length; index += 1) {
    const prev = filtered[filtered.length - 1];
    const point = points[index];
    const dx = point.x - prev.x;
    const dy = point.y - prev.y;
    if (Math.hypot(dx, dy) > eps) filtered.push(point);
  }
  return filtered;
}

function resamplePolylineDensity(points, numSamples, alpha) {
  const polyline = removeDuplicatePoints(points);
  if (polyline.length < 2) return [];
  const segLen = [];
  for (let index = 0; index < polyline.length - 1; index += 1) {
    segLen.push(Math.hypot(polyline[index + 1].x - polyline[index].x, polyline[index + 1].y - polyline[index].y));
  }
  const maxLen = Math.max(...segLen);
  const bias = maxLen * (Math.exp(3 * alpha) - 1) / 2;
  const weights = segLen.map((length) => length + bias);
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
  const cdf = [0];
  weights.forEach((weight) => cdf.push(cdf[cdf.length - 1] + weight / weightSum));
  const samples = [];
  for (let sample = 0; sample < numSamples; sample += 1) {
    const u = numSamples === 1 ? 0 : sample / (numSamples - 1);
    let segIndex = 0;
    while (segIndex < cdf.length - 2 && cdf[segIndex + 1] < u) segIndex += 1;
    const localT = (u - cdf[segIndex]) / (cdf[segIndex + 1] - cdf[segIndex] + 1e-6);
    const start = polyline[segIndex];
    const end = polyline[segIndex + 1];
    samples.push({
      x: start.x * (1 - localT) + end.x * localT,
      y: start.y * (1 - localT) + end.y * localT
    });
  }
  return samples;
}

function trajectorySampleAlpha() {
  return Number(trajectoryAlphaInput.value);
}

function trajectoryFrameCount() {
  return Number(trajectoryFramesInput.value);
}

function processedTrajectoryPoints() {
  return resamplePolylineDensity(scaledTrajectoryPoints(), trajectoryFrameCount(), trajectorySampleAlpha());
}

function alternatingColor(index, evenColor, oddColor) {
  return index % 2 === 0 ? evenColor : oddColor;
}

function setStatus(text, isGenerating = false) {
  statusText.textContent = text;
  statusText.classList.toggle("generating", isGenerating);
}

function setGenerationProgress(value) {
  const percent = Math.max(0, Math.min(100, Math.round(value)));
  generationProgressBar.style.width = `${percent}%`;
  generationProgressText.textContent = `${percent}%`;
}

function startGenerationProgress() {
  window.clearInterval(generationProgressTimer);
  window.clearTimeout(generationProgressHideTimer);
  const startedAt = performance.now();
  generateBtn.disabled = true;
  generationProgress.classList.remove("hidden");
  generationProgress.classList.add("active");
  requestAnimationFrame(alignViewerHeight);
  setGenerationProgress(3);
  setStatus("diffusion 3%", true);
  generationProgressTimer = window.setInterval(() => {
    const elapsed = performance.now() - startedAt;
    const percent = 3 + 89 * (1 - Math.exp(-elapsed / 8500));
    setGenerationProgress(percent);
    setStatus(`diffusion ${Math.round(Math.min(92, percent))}%`, true);
  }, 240);
}

function finishGenerationProgress() {
  window.clearInterval(generationProgressTimer);
  generationProgressTimer = 0;
  generationProgress.classList.remove("active");
  setGenerationProgress(100);
  setStatus("complete");
  generateBtn.disabled = false;
  generationProgressHideTimer = window.setTimeout(() => {
    generationProgress.classList.add("hidden");
    setGenerationProgress(0);
    requestAnimationFrame(alignViewerHeight);
  }, 900);
}

function failGenerationProgress(message = "generation failed") {
  window.clearInterval(generationProgressTimer);
  window.clearTimeout(generationProgressHideTimer);
  generationProgressTimer = 0;
  generationProgress.classList.remove("active");
  requestAnimationFrame(alignViewerHeight);
  setStatus(message);
  generateBtn.disabled = false;
}

async function readResponseError(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const errorData = await response.json();
    return errorData.detail || response.statusText;
  }
  const text = await response.text();
  return text.trim() || response.statusText;
}

function drawFramePath(ctx, points, evenColor, oddColor, width) {
  if (points.length < 2) return;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "butt";
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    ctx.strokeStyle = alternatingColor(index, evenColor, oddColor);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }
}

function updateTrajectoryScaleLabel() {
  trajectoryScaleValue.value = `${trajectoryScale.toFixed(2)}x`;
}

function drawCarReference() {
  if (!carReference.complete) return;
  const width = carLengthMeters * trajectoryPixelsPerMeter;
  const height = width * carReference.naturalHeight / carReference.naturalWidth;
  const x = 20;
  const y = 0;
  trajectoryCtx.save();
  trajectoryCtx.globalAlpha = 0.09;
  trajectoryCtx.drawImage(carReference, x, y, width, height);
  trajectoryCtx.restore();
}

function drawTrajectoryCanvas() {
  trajectoryCtx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
  trajectoryCtx.fillStyle = "#ffffff";
  trajectoryCtx.fillRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
  trajectoryCtx.strokeStyle = "#e5e7eb";
  trajectoryCtx.lineWidth = 1;
  for (let x = 40; x < trajectoryCanvas.width; x += 40) {
    trajectoryCtx.beginPath();
    trajectoryCtx.moveTo(x, 0);
    trajectoryCtx.lineTo(x, trajectoryCanvas.height);
    trajectoryCtx.stroke();
  }
  for (let y = 40; y < trajectoryCanvas.height; y += 40) {
    trajectoryCtx.beginPath();
    trajectoryCtx.moveTo(0, y);
    trajectoryCtx.lineTo(trajectoryCanvas.width, y);
    trajectoryCtx.stroke();
  }
  drawCarReference();
  const processedTrajectory = processedTrajectoryPoints();
  drawFramePath(trajectoryCtx, processedTrajectory, "#0f766e", "#dc2626", 5);
  keyframes.forEach((keyframe) => {
    const p = processedTrajectory[Math.round(keyframe.t * (processedTrajectory.length - 1))];
    if (!p) return;
    trajectoryCtx.fillStyle = "#b45309";
    trajectoryCtx.beginPath();
    trajectoryCtx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    trajectoryCtx.fill();
  });
}

function stickmanEnabled() {
  return false;
}

function updateStickmanPanel() {
  stickmanPanel.classList.add("hidden");
  keyframes = [];
  poseLines = [];
  currentPoseLine = [];
  renderKeyframes();
  progressMarkers.innerHTML = "";
  drawTrajectoryCanvas();
  requestAnimationFrame(alignViewerHeight);
}

function drawPoseCanvas() {
  poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  poseCtx.fillStyle = "#fff";
  poseCtx.fillRect(0, 0, poseCanvas.width, poseCanvas.height);
  poseCtx.strokeStyle = "#eef1ee";
  poseCtx.lineWidth = 1;
  poseCtx.beginPath();
  poseCtx.moveTo(poseCanvas.width / 2, 0);
  poseCtx.lineTo(poseCanvas.width / 2, poseCanvas.height);
  poseCtx.moveTo(0, poseCanvas.height / 2);
  poseCtx.lineTo(poseCanvas.width, poseCanvas.height / 2);
  poseCtx.stroke();
  [...poseLines, currentPoseLine].forEach((line, lineIndex) => {
    if (line.length < 2) return;
    poseCtx.strokeStyle = lineIndex % 2 === 0 ? "#17211d" : "#0f766e";
    poseCtx.lineWidth = 4;
    poseCtx.lineJoin = "round";
    poseCtx.lineCap = "round";
    poseCtx.beginPath();
    line.forEach((point, index) => {
      if (index === 0) poseCtx.moveTo(point.x, point.y);
      else poseCtx.lineTo(point.x, point.y);
    });
    poseCtx.stroke();
  });
  requestAnimationFrame(alignViewerHeight);
}

function setTrajectory(points) {
  trajectory = points;
  keyframes = [];
  renderKeyframes();
  drawTrajectoryCanvas();
}

function setPoseTemplate(name) {
  const c = poseCanvas.width / 2;
  const cy = poseCanvas.height / 2;
  const templates = {
    walk: [
      [[c - 26, cy - 112], [c + 26, cy - 112], [c + 28, cy - 82], [c - 28, cy - 82], [c - 26, cy - 112]],
      [[c, cy - 78], [c, cy + 18]],
      [[c - 4, cy - 44], [c - 62, cy - 8], [c - 86, cy + 34]],
      [[c + 4, cy - 40], [c + 64, cy - 16], [c + 78, cy + 28]],
      [[c - 2, cy + 18], [c - 48, cy + 88], [c - 80, cy + 132]],
      [[c + 2, cy + 18], [c + 48, cy + 92], [c + 82, cy + 132]]
    ],
    hand: [
      [[c - 26, cy - 118], [c + 26, cy - 118], [c + 28, cy - 88], [c - 28, cy - 88], [c - 26, cy - 118]],
      [[c, cy - 84], [c, cy + 20]],
      [[c - 5, cy - 52], [c - 52, cy - 116], [c - 70, cy - 158]],
      [[c + 5, cy - 50], [c + 62, cy - 10], [c + 94, cy + 28]],
      [[c - 2, cy + 20], [c - 44, cy + 92], [c - 48, cy + 142]],
      [[c + 2, cy + 20], [c + 46, cy + 90], [c + 52, cy + 142]]
    ]
  };
  poseLines = templates[name].map((line) => line.map(([x, y]) => ({ x, y })));
  currentPoseLine = [];
  drawPoseCanvas();
}

function renderKeyframes() {
  keyframeList.innerHTML = "";
  keyframes
    .map((keyframe, index) => ({ keyframe, index }))
    .sort((a, b) => a.keyframe.t - b.keyframe.t)
    .forEach(({ keyframe, index }) => {
      const chip = document.createElement("span");
      chip.className = "keyframe-chip";
      chip.textContent = `${Math.round(keyframe.t * 100)}%`;
      chip.addEventListener("click", () => {
        restoreKeyframe(index);
      });
      const button = document.createElement("button");
      button.textContent = "x";
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        keyframes.splice(index, 1);
        renderKeyframes();
        drawTrajectoryCanvas();
      });
      chip.appendChild(button);
      keyframeList.appendChild(chip);
    });
}

trajectoryCanvas.addEventListener("pointerdown", (event) => {
  drawingTrajectory = true;
  trajectory = [canvasPoint(event, trajectoryCanvas)];
  keyframes = [];
  renderKeyframes();
  drawTrajectoryCanvas();
});

trajectoryCanvas.addEventListener("pointermove", (event) => {
  if (!drawingTrajectory) return;
  trajectory.push(canvasPoint(event, trajectoryCanvas));
  drawTrajectoryCanvas();
});

window.addEventListener("pointerup", () => {
  drawingTrajectory = false;
  drawingPose = false;
  if (currentPoseLine.length > 1 && poseLines.length < 6) {
    poseLines.push(currentPoseLine);
  }
  currentPoseLine = [];
  drawPoseCanvas();
});

poseCanvas.addEventListener("pointerdown", (event) => {
  if (poseLines.length >= 6) return;
  drawingPose = true;
  currentPoseLine = [canvasPoint(event, poseCanvas)];
});

poseCanvas.addEventListener("pointermove", (event) => {
  if (!drawingPose) return;
  currentPoseLine.push(canvasPoint(event, poseCanvas));
  drawPoseCanvas();
});

document.getElementById("clearTrajectoryBtn").addEventListener("click", () => setTrajectory([]));
trajectoryScaleInput.addEventListener("input", () => {
  trajectoryScale = Number(trajectoryScaleInput.value);
  updateTrajectoryScaleLabel();
  drawTrajectoryCanvas();
});
function redrawTrajectoryFromParameters() {
  requestAnimationFrame(drawTrajectoryCanvas);
}
[trajectoryAlphaInput, trajectoryFramesInput].forEach((input) => {
  input.addEventListener("input", redrawTrajectoryFromParameters);
  input.addEventListener("change", redrawTrajectoryFromParameters);
  input.addEventListener("keyup", redrawTrajectoryFromParameters);
  input.addEventListener("blur", redrawTrajectoryFromParameters);
});
document.getElementById("clearPoseBtn").addEventListener("click", () => {
  poseLines = [];
  currentPoseLine = [];
  drawPoseCanvas();
});
document.getElementById("templateWalkBtn").addEventListener("click", () => setPoseTemplate("walk"));
document.getElementById("templateHandBtn").addEventListener("click", () => setPoseTemplate("hand"));
document.getElementById("exampleMBtn").addEventListener("click", () => {
  const pts = [];
  for (let i = 0; i < 180; i++) {
    const t = i / 179;
    const x = 70 + t * 500;
    const y = 290 - Math.sin(t * Math.PI * 4) * 82 * (1 - 0.25 * t);
    pts.push({ x, y });
  }
  setTrajectory(pts);
});
document.getElementById("exampleArcBtn").addEventListener("click", () => {
  const pts = [];
  for (let i = 0; i < 180; i++) {
    const t = i / 179;
    pts.push({ x: 82 + t * 500, y: 320 - Math.sin(t * Math.PI) * 210 });
  }
  setTrajectory(pts);
});
document.getElementById("exampleNarutoBtn").addEventListener("click", () => {
  const pts = [];
  for (let i = 0; i < 220; i++) {
    const t = i / 219;
    const r = 18 + t * 185;
    const a = t * Math.PI * 5.2;
    pts.push({ x: 320 + Math.cos(a) * r, y: 215 + Math.sin(a) * r * 0.72 });
  }
  setTrajectory(pts);
});

document.getElementById("savePoseBtn").addEventListener("click", () => {
  saveCurrentPose();
});

const viewer = document.getElementById("viewer");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111917);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
let cameraYaw = 0;
let cameraPitch = 0.52;
let cameraDistance = 11;
let draggingView = false;
let lastViewPoint = null;
function updateCamera() {
  const x = Math.sin(cameraYaw) * Math.cos(cameraPitch) * cameraDistance;
  const y = Math.sin(cameraPitch) * cameraDistance;
  const z = Math.cos(cameraYaw) * Math.cos(cameraPitch) * cameraDistance;
  camera.position.set(cameraTarget.x + x, cameraTarget.y + y, cameraTarget.z + z);
  camera.lookAt(cameraTarget);
}
updateCamera();
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewer.appendChild(renderer.domElement);
const skeletonGroup = new THREE.Group();
const trajGroup = new THREE.Group();
scene.add(skeletonGroup);
scene.add(trajGroup);
scene.add(new THREE.HemisphereLight(0xffffff, 0x22302b, 1.6));
const grid = new THREE.GridHelper(12, 12, 0x315048, 0x20302b);
scene.add(grid);

function resizeViewer() {
  const rect = viewer.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}

function alignViewerHeight() {
  const rightPanel = document.querySelector(".right-panel");
  const resultRow = document.querySelector(".result-row");
  if (!trajectoryCanvas || !rightPanel) {
    resizeViewer();
    return;
  }
  const targetBottom = trajectoryCanvas.getBoundingClientRect().bottom;
  const viewerTop = viewer.getBoundingClientRect().top;
  const viewerBottom = viewer.getBoundingClientRect().bottom;
  const rightPanelTop = rightPanel.getBoundingClientRect().top;
  if (rightPanelTop > targetBottom || window.innerWidth <= 1080) {
    viewer.style.height = "";
    resizeViewer();
    return;
  }
  const belowViewerHeight = resultRow ? Math.max(0, resultRow.getBoundingClientRect().bottom - viewerBottom) : 0;
  const availableHeight = window.innerHeight - viewerTop - belowViewerHeight - 22;
  const alignedHeight = targetBottom - viewerTop - belowViewerHeight;
  viewer.style.height = `${Math.max(320, Math.min(alignedHeight, availableHeight))}px`;
  resizeViewer();
}

window.addEventListener("resize", alignViewerHeight);
window.addEventListener("load", alignViewerHeight);
requestAnimationFrame(alignViewerHeight);
requestAnimationFrame(() => requestAnimationFrame(alignViewerHeight));
setTimeout(alignViewerHeight, 100);
setTimeout(alignViewerHeight, 300);
setTimeout(alignViewerHeight, 800);

if (window.ResizeObserver) {
  const layoutObserver = new ResizeObserver(() => alignViewerHeight());
  layoutObserver.observe(document.querySelector(".left-panel"));
  layoutObserver.observe(document.querySelector(".right-panel"));
  layoutObserver.observe(poseCanvas);
  layoutObserver.observe(trajectoryCanvas);
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.traverse?.((node) => {
      node.geometry?.dispose();
      if (Array.isArray(node.material)) node.material.forEach((material) => material.dispose());
      else node.material?.dispose();
    });
  }
}

function bodyScale() {
  return sceneTransform.scale;
}

function rootVector(root, yOffset = 0) {
  return new THREE.Vector3(
    (root[0] - sceneTransform.cx) * sceneTransform.scale,
    yOffset,
    -(root[2] - sceneTransform.cz) * sceneTransform.scale
  );
}

function jointVector(point, root) {
  const rootPos = rootVector(root);
  const s = bodyScale();
  return new THREE.Vector3(
    rootPos.x + (point[0] - root[0]) * s,
    (point[1] - root[1]) * s + 0.85,
    rootPos.z - (point[2] - root[2]) * s
  );
}

function updateSceneTransform(data, span = 7.2) {
  const paths = [data.input_trajectory, data.pred_trajectory].filter(Boolean);
  const flat = paths.flat();
  const xs = flat.map((p) => p[0]);
  const zs = flat.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  sceneTransform = {
    cx: (minX + maxX) / 2,
    cz: (minZ + maxZ) / 2,
    scale: span / Math.max(maxX - minX, maxZ - minZ, 1e-6)
  };
}

function trajectoryVector(p, yOffset) {
  return new THREE.Vector3(
    (p[0] - sceneTransform.cx) * sceneTransform.scale,
    yOffset,
    -(p[1] - sceneTransform.cz) * sceneTransform.scale
  );
}

function pathVectors(points, yOffset) {
  if (!points || points.length < 2) return [];
  return points.map((p) => trajectoryVector(p, yOffset));
}

function makeLine(points, color, width = 2) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, linewidth: width });
  return new THREE.Line(geometry, material);
}

function alternatingThreeColor(index, evenColor, oddColor) {
  return new THREE.Color(index % 2 === 0 ? evenColor : oddColor);
}

function makeFramePath3D(points, yOffset, evenColor, oddColor) {
  const vectors = pathVectors(points, yOffset);
  const group = new THREE.Group();
  for (let index = 0; index < vectors.length - 1; index += 1) {
    const color = alternatingThreeColor(index, evenColor, oddColor);
    group.add(makeLine([vectors[index], vectors[index + 1]], color));
  }
  return group;
}

function makeLimb(start, end, radius, color) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  if (length < 1e-5) return null;
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 10);
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.72 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return mesh;
}

function currentRootVector(frameIndex) {
  if (!result) return new THREE.Vector3(0, 1, 0);
  const joints = result.pred_joint[frameIndex % result.motion_length];
  const root = joints[0];
  const rootPos = rootVector(root);
  return new THREE.Vector3(rootPos.x, 1, rootPos.z);
}

function focusActorFront(frameIndex) {
  cameraTarget.copy(currentRootVector(frameIndex));
  cameraYaw = 0;
  cameraPitch = 0.18;
  cameraDistance = 4.2;
  updateCamera();
}

function resetCamera() {
  cameraTarget.set(0, 1, 0);
  cameraYaw = 0;
  cameraPitch = 0.52;
  cameraDistance = 11;
  updateCamera();
}

function stickmanFrames() {
  return (result?.stickman_index || []).slice().sort((a, b) => a - b);
}

function loadResult(data) {
  result = data;
  frame = 0;
  progressSlider.disabled = false;
  progressSlider.max = String(data.motion_length - 1);
  progressSlider.value = "0";
  clearGroup(trajGroup);
  updateSceneTransform(data);
  trajGroup.add(makeFramePath3D(data.input_trajectory, 0.03, 0x0f766e, 0xdc2626));
  trajGroup.add(makeFramePath3D(data.pred_trajectory, 0.06, 0x2dd4bf, 0x2563eb));
  const frames = stickmanFrames();
  statusText.textContent = frames.length > 0 ? `stickman frames ${frames.join(", ")}` : "complete";
  renderProgressMarkers();
}

function renderSkeleton() {
  clearGroup(skeletonGroup);
  if (!result) return;
  const frameIndex = frame % result.motion_length;
  const joints = result.pred_joint[frameIndex];
  const root = joints[0];
  const pairs = result.joints_num === 21 ? kitPairs : t2mPairs;
  const material = new THREE.MeshStandardMaterial({ color: 0xe8d0bf, roughness: 0.72 });
  joints.forEach((joint, index) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(index === 0 ? 0.038 : 0.028, 12, 12), material);
    mesh.position.copy(jointVector(joint, root));
    skeletonGroup.add(mesh);
  });
  pairs.forEach(([a, b]) => {
    const limb = makeLimb(jointVector(joints[a], root), jointVector(joints[b], root), 0.014, 0xf8dec9);
    if (limb) skeletonGroup.add(limb);
  });
  frameText.textContent = `frame ${frameIndex}`;
  if (!draggingProgress) progressSlider.value = String(frameIndex);
}

function animate(time) {
  requestAnimationFrame(animate);
  if (playing && result && time - lastFrameTime >= playbackFrameMs) {
    frame = (frame + 1) % result.motion_length;
    lastFrameTime = time;
  }
  renderSkeleton();
  renderer.render(scene, camera);
}
animate();

document.getElementById("playPauseBtn").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("playPauseBtn").textContent = playing ? "Pause" : "Play";
  statusText.textContent = playing ? "loop playing" : "paused";
});
document.getElementById("resetViewBtn").addEventListener("click", () => {
  resetCamera();
});
progressSlider.addEventListener("input", () => {
  if (!result) return;
  draggingProgress = true;
  playing = false;
  document.getElementById("playPauseBtn").textContent = "Play";
  frame = Number(progressSlider.value);
  frameText.textContent = `frame ${frame}`;
});
progressSlider.addEventListener("change", () => {
  draggingProgress = false;
});
viewer.addEventListener("pointerdown", (event) => {
  draggingView = true;
  lastViewPoint = { x: event.clientX, y: event.clientY };
});
window.addEventListener("pointermove", (event) => {
  if (!draggingView) return;
  const dx = event.clientX - lastViewPoint.x;
  const dy = event.clientY - lastViewPoint.y;
  cameraYaw -= dx * 0.008;
  cameraPitch = Math.max(0.08, Math.min(1.25, cameraPitch + dy * 0.006));
  lastViewPoint = { x: event.clientX, y: event.clientY };
  updateCamera();
});
window.addEventListener("pointerup", () => {
  draggingView = false;
});
viewer.addEventListener("wheel", (event) => {
  event.preventDefault();
  cameraDistance = Math.max(4, Math.min(22, cameraDistance + event.deltaY * 0.01));
  updateCamera();
});

generateBtn.addEventListener("click", async () => {
  const generationTrajectory = scaledTrajectoryPoints();
  if (generationTrajectory.length < 2) {
    setStatus("draw trajectory");
    return;
  }
  if (stickmanEnabled() && poseLines.length === 6) {
    saveCurrentPose({ silent: true });
  }
  const payload = {
    text: textPrompt.value,
    trajectory: generationTrajectory,
    length: trajectoryFrameCount(),
    density: trajectorySampleAlpha(),
    trajectory_scale: trajectoryPixelsPerMeter,
    ifg_repeat: Number(ifgRepeatInput.value),
    ifg_scale: Number(ifgScaleInput.value)
  };
  startGenerationProgress();
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      throw new Error(await readResponseError(response));
    }
    const data = await response.json();
    loadResult(data);
    finishGenerationProgress();
  } catch (error) {
    failGenerationProgress(error.message);
  }
});

updateTrajectoryScaleLabel();
drawTrajectoryCanvas();
drawPoseCanvas();
setPoseTemplate("walk");
updateStickmanPanel();

function saveCurrentPose(options = {}) {
  if (!stickmanEnabled()) return false;
  if (poseLines.length !== 6) {
    if (!options.silent) statusText.textContent = "draw 6 strokes";
    return false;
  }
  keyframes.push({ t: Number(poseTime.value) / 100, lines: poseLines.map((line) => line.map((p) => ({ ...p }))) });
  poseLines = [];
  renderKeyframes();
  drawPoseCanvas();
  drawTrajectoryCanvas();
  if (!options.silent) statusText.textContent = "key pose saved";
  return true;
}

function restoreKeyframe(index) {
  const keyframe = keyframes[index];
  if (!keyframe) return;
  poseTime.value = Math.round(keyframe.t * 100);
  poseLines = keyframe.lines.map((line) => line.map((point) => ({ ...point })));
  currentPoseLine = [];
  drawPoseCanvas();
  statusText.textContent = `restored ${Math.round(keyframe.t * 100)}%`;
}

function keyframeIndexForFrame(frameIndex) {
  const frames = stickmanFrames();
  const sortedKeyframes = keyframes
    .map((keyframe, index) => ({ keyframe, index }))
    .sort((a, b) => a.keyframe.t - b.keyframe.t);
  const position = frames.indexOf(frameIndex);
  return sortedKeyframes[position]?.index;
}

function jumpToStickmanFrame(frameIndex) {
  if (!result) return;
  frame = frameIndex;
  progressSlider.value = String(frameIndex);
  playing = false;
  document.getElementById("playPauseBtn").textContent = "Play";
  focusActorFront(frameIndex);
  const keyframeIndex = keyframeIndexForFrame(frameIndex);
  if (keyframeIndex !== undefined) restoreKeyframe(keyframeIndex);
  statusText.textContent = `stickman frame ${frameIndex}`;
}

function renderProgressMarkers() {
  progressMarkers.innerHTML = "";
  const frames = stickmanFrames();
  if (!result || frames.length === 0) return;
  const maxFrame = Math.max(1, result.motion_length - 1);
  frames.forEach((frameIndex) => {
    const marker = document.createElement("button");
    marker.className = "progress-marker";
    marker.type = "button";
    marker.style.left = `${(frameIndex / maxFrame) * 100}%`;
    marker.title = `Stickman frame ${frameIndex}`;
    marker.addEventListener("click", () => jumpToStickmanFrame(frameIndex));
    progressMarkers.appendChild(marker);
  });
}
