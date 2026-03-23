/**
 * FoodScan 3D — Main Application Entry Point
 * Orchestrates the upload → processing → 3D viewer flow
 */

import './style.css';
import { SceneManager } from './scene.js';
import { segmentIngredients, correctPlateOrientation } from './segmentation.js';
import { generateMesh, highlightSegment } from './mesh-generator.js';
import { calculateVolumes, formatVolume, estimateCalories } from './volume-calculator.js';
import { checkBackend, reconstructWithBackend, getGlbUrl, isBackendAvailable } from './backend.js';

// ---- State ----
const state = {
  images: [],
  depthWorker: null,
  sceneManager: null,
  foodGroup: null,
  segments: [],
  volumes: {},
  plateDiameter: 26,
  showSegments: true,
  showWireframe: false,
  backendGlbLoaded: false, // true when backend GLB is already in the scene
  // Cached data for tuning panel regeneration
  cachedImageData: null,
  cachedDepthResult: null,
  cachedModelName: 'depth-anything-v2-small',
  cachedClipSegResult: null,
};

// ---- DOM Elements ----
const $ = (id) => document.getElementById(id);

const screens = {
  upload: $('upload-screen'),
  processing: $('processing-screen'),
  viewer: $('viewer-screen'),
};

// ---- Screen Management ----
function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove('active'));
  screens[name].classList.add('active');
}

// ---- Image Upload ----
function initUpload() {
  const dropZone = $('drop-zone');
  const fileInput = $('file-input');
  const thumbnails = $('thumbnails');
  const scaleSection = $('scale-section');
  const processBtn = $('process-btn');

  // Click to browse
  dropZone.addEventListener('click', () => fileInput.click());

  // Drag & drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
  });

  function handleFiles(files) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      if (state.images.some(img => img.name === file.name)) continue;
      
      const url = URL.createObjectURL(file);
      state.images.push({ file, url, name: file.name });
    }
    updateThumbnails();
  }

  function updateThumbnails() {
    if (state.images.length === 0) {
      thumbnails.classList.add('hidden');
      scaleSection.classList.add('hidden');
      processBtn.classList.add('hidden');
      return;
    }

    thumbnails.classList.remove('hidden');
    scaleSection.classList.remove('hidden');
    processBtn.classList.remove('hidden');
    thumbnails.innerHTML = '';

    state.images.forEach((img, idx) => {
      const item = document.createElement('div');
      item.className = 'thumb-item';
      item.innerHTML = `
        <img src="${img.url}" alt="${img.name}" />
        <button class="thumb-remove" data-idx="${idx}">×</button>
      `;
      thumbnails.appendChild(item);
    });

    // Remove buttons
    thumbnails.querySelectorAll('.thumb-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.idx);
        URL.revokeObjectURL(state.images[idx].url);
        state.images.splice(idx, 1);
        updateThumbnails();
      });
    });
  }

  // Plate diameter
  $('plate-diameter').addEventListener('change', (e) => {
    state.plateDiameter = parseFloat(e.target.value) || 26;
  });

  // Process button
  processBtn.addEventListener('click', () => {
    if (state.images.length === 0) return;
    startProcessing();
  });
}

// ---- Processing Pipeline ----
async function startProcessing() {
  showScreen('processing');
  
  const progressBar = $('progress-bar');
  const statusText = $('processing-status');
  const titleText = $('processing-title');

  function setProgress(pct, status) {
    progressBar.style.width = `${pct}%`;
    statusText.textContent = status;
  }

  // ── Backend-first path (MapAnything) ──────────────────────────────────────
  if (isBackendAvailable()) {
    try {
      titleText.textContent = 'Metric 3D Reconstruction...';
      setProgress(5, 'Connecting to local AI engine...');

      const imageFiles = state.images.map(img => img.file).filter(Boolean);

      const result = await reconstructWithBackend(
        imageFiles,
        state.plateDiameter,
        (msg) => setProgress(50, msg)
      );

      setProgress(80, 'Building 3D scene...');

      // Load the GLB from the backend
      const glbUrl = getGlbUrl(result.glb_url);
      if (!state.sceneManager) {
        const canvas = $('three-canvas');
        state.sceneManager = new SceneManager(canvas);
      }
      await state.sceneManager.loadGlb(glbUrl);
      state.sceneManager.resize();

      // Populate state from backend response
      state.segments = result.segments.map(s => ({
        id: s.id,
        label: s.label,
        color: s.color,
        percentage: s.percentage || 0,
      }));
      state.volumes = {};
      result.segments.forEach(s => { state.volumes[s.id] = s.volume_ml; });

      setProgress(100, 'Done!');
      titleText.textContent = 'Complete!';
      await sleep(400);
      state.backendGlbLoaded = true;
      showViewer();
      return; // Done — skip browser pipeline
    } catch (err) {
      console.warn('Backend failed, falling back to browser pipeline:', err);
      setProgress(0, 'Falling back to browser AI...');
      titleText.textContent = 'Analyzing Food...';
    }
  }

  // ── Browser-only fallback pipeline ───────────────────────────────────────
  const steps = $('processing-steps').querySelectorAll('.step');

  function setStep(stepName, state) {
    steps.forEach(s => {
      if (s.dataset.step === stepName) {
        s.classList.remove('active', 'done');
        s.classList.add(state);
      }
    });
  }
  
  try {
    // Step 1: Load and prepare image
    setStep('depth', 'active');
    setProgress(5, 'Loading image...');
    titleText.textContent = 'Analyzing Food...';
    
    const img = state.images[0]; // Use first image for now
    const imageData = await loadImageData(img.url);
    state.cachedImageData = imageData;
    
    // Step 2: Depth estimation + CLIPSeg in parallel
    setProgress(10, 'Loading AI models (first time may take 30-60s)...');
    
    const depthPromise = runDepthEstimation(imageData, (progress) => {
      setProgress(10 + progress * 35, 'Running depth estimation...');
    });
    
    const clipSegPromise = runClipSegWorker(imageData).catch(err => {
      console.warn('CLIPSeg failed, will use depth-based masking:', err);
      return null;
    });
    
    const [depthResult, clipSegResult] = await Promise.all([depthPromise, clipSegPromise]);
    
    state.cachedDepthResult = depthResult;
    state.cachedClipSegResult = clipSegResult;
    setStep('depth', 'done');
    setProgress(50, 'AI analysis complete');
    
    // Step 3: Segmentation
    setStep('segment', 'active');
    setProgress(55, 'Segmenting ingredients...');
    
    await sleep(100); // Allow UI to update
    
    // Apply plate orientation correction if auto-level is checked
    let depthForSeg = depthResult.depth;
    const autoLevel = $('tune-autolevel') && $('tune-autolevel').checked;
    
    const segResult = segmentIngredients(
      imageData.data,
      imageData.width,
      imageData.height,
      depthForSeg,
      depthResult.width,
      depthResult.height,
      6,
      0.15,
      clipSegResult ? 'clipseg' : 'histogram',
      clipSegResult
    );
    
    state.segments = segResult.segments;
    setStep('segment', 'done');
    setProgress(70, 'Segmentation complete');
    
    // Step 4: Mesh generation
    setStep('mesh', 'active');
    setProgress(75, 'Generating 3D mesh...');
    
    await sleep(100);
    
    const foodGroup = generateMesh(segResult, imageData.data, state.plateDiameter);
    state.foodGroup = foodGroup;
    
    setStep('mesh', 'done');
    setProgress(85, '3D mesh generated');
    
    // Step 5: Volume calculation
    setStep('volume', 'active');
    setProgress(90, 'Calculating volumes...');
    
    await sleep(100);
    
    const volumes = calculateVolumes(foodGroup);
    state.volumes = volumes;
    
    setStep('volume', 'done');
    setProgress(100, 'Done!');
    titleText.textContent = 'Complete!';
    
    await sleep(500);
    
    // Show viewer
    showViewer();
    
  } catch (error) {
    console.error('Processing error:', error);
    statusText.textContent = `Error: ${error.message}`;
    titleText.textContent = 'Processing Failed';
    progressBar.style.background = '#f43f5e';
  }
}

async function loadImageData(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    // Only set crossOrigin for non-blob URLs (blob URLs are same-origin)
    if (!url.startsWith('blob:')) {
      img.crossOrigin = 'anonymous';
    }
    img.onload = () => {
      // Limit size for processing
      const maxSize = 512;
      let w = img.width, h = img.height;
      if (w > maxSize || h > maxSize) {
        const scale = maxSize / Math.max(w, h);
        w = Math.floor(w * scale);
        h = Math.floor(h * scale);
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h);
      resolve(data);
    };
    img.onerror = (e) => reject(new Error('Failed to load image: ' + (e.message || url)));
    img.src = url;
  });
}

async function runDepthEstimation(imageData, onProgress, modelName = 'depth-anything-v2-small') {
  return new Promise((resolve, reject) => {
    // Create worker
    const worker = new Worker(
      new URL('./workers/depth-worker.js', import.meta.url),
      { type: 'module' }
    );
    
    state.depthWorker = worker;
    
    worker.onmessage = (e) => {
      const { type } = e.data;
      
      if (type === 'status') {
        console.log('Depth worker:', e.data.message);
      } else if (type === 'model-progress') {
        const pct = e.data.total > 0 ? e.data.loaded / e.data.total : 0;
        onProgress(Math.min(pct, 0.9));
      } else if (type === 'result') {
        onProgress(1);
        worker.terminate();
        resolve({
          depth: e.data.depth,
          width: e.data.width,
          height: e.data.height,
        });
      } else if (type === 'error') {
        worker.terminate();
        reject(new Error(e.data.message));
      }
    };
    
    worker.onerror = (error) => {
      worker.terminate();
      reject(error);
    };
    
    // Send image data to worker
    const buffer = imageData.data.buffer.slice(0);
    worker.postMessage({
      type: 'estimate',
      imageData: buffer,
      width: imageData.width,
      height: imageData.height,
      modelName: modelName,
    }, [buffer]);
  });
}

async function runClipSegWorker(imageData) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('./workers/seg-worker.js', import.meta.url),
      { type: 'module' }
    );
    
    worker.onmessage = (e) => {
      const { type } = e.data;
      
      if (type === 'status') {
        console.log('CLIPSeg worker:', e.data.message);
      } else if (type === 'result') {
        worker.terminate();
        resolve({
          mask: new Float32Array(e.data.mask),
          maskWidth: e.data.maskWidth,
          maskHeight: e.data.maskHeight,
        });
      } else if (type === 'error') {
        worker.terminate();
        reject(new Error(e.data.message));
      }
    };
    
    worker.onerror = (error) => {
      worker.terminate();
      reject(error);
    };
    
    // Send image data to worker
    const buffer = imageData.data.buffer.slice(0);
    worker.postMessage({
      type: 'segment',
      imageData: buffer,
      width: imageData.width,
      height: imageData.height,
    }, [buffer]);
  });
}

// ---- 3D Viewer ----
function showViewer() {
  showScreen('viewer');
  
  // Initialize scene
  if (!state.sceneManager) {
    const canvas = $('three-canvas');
    state.sceneManager = new SceneManager(canvas);
  }
  
  // Only set browser-generated mesh if not using backend GLB
  if (!state.backendGlbLoaded && state.foodGroup) {
    state.sceneManager.setFoodModel(state.foodGroup);
  }
  state.sceneManager.resize();
  
  // Setup click callback
  state.sceneManager.onSegmentClick = (segId) => {
    selectSegment(segId);
  };
  
  // Populate sidebar
  populateSidebar();
  
  // Setup viewer controls
  initViewerControls();
  
  // Setup tuning panel
  initTuningPanel();
}

function populateSidebar() {
  const list = $('ingredient-list');
  const totalVol = $('total-vol');
  
  // Calculate total
  let total = 0;
  state.segments.forEach(s => {
    total += state.volumes[s.id] || 0;
  });
  
  const totalFormatted = formatVolume(total);
  totalVol.textContent = `${totalFormatted.ml} ml`;
  
  // Create ingredient items
  list.innerHTML = '';
  state.segments.forEach(seg => {
    const vol = state.volumes[seg.id] || 0;
    const formatted = formatVolume(vol);
    const cal = estimateCalories(vol, seg.label);
    
    const item = document.createElement('div');
    item.className = 'ingredient-item';
    item.dataset.segId = seg.id;
    item.innerHTML = `
      <div class="ingredient-swatch" style="background: ${seg.color}"></div>
      <div class="ingredient-info">
        <div class="ingredient-name">${seg.label}</div>
        <div class="ingredient-vol">${formatted.ml} ml · ~${cal} kcal</div>
      </div>
      <div class="ingredient-pct">${seg.percentage}%</div>
    `;
    
    item.addEventListener('click', () => {
      selectSegment(seg.id);
    });
    
    list.appendChild(item);
  });
}

function selectSegment(segId) {
  const list = $('ingredient-list');
  const selectedInfo = $('selected-info');
  const selectedDetail = $('selected-detail');
  
  // Update list selection
  list.querySelectorAll('.ingredient-item').forEach(item => {
    const itemSeg = parseInt(item.dataset.segId);
    item.classList.toggle('selected', itemSeg === segId);
  });
  
  // Update 3D highlight
  if (state.foodGroup) {
    highlightSegment(state.foodGroup, segId);
  }
  
  if (segId === null || segId === undefined) {
    selectedInfo.classList.add('hidden');
    return;
  }
  
  // Show detail
  const seg = state.segments.find(s => s.id === segId);
  if (!seg) return;
  
  const vol = state.volumes[seg.id] || 0;
  const formatted = formatVolume(vol);
  const cal = estimateCalories(vol, seg.label);
  
  selectedInfo.classList.remove('hidden');
  selectedDetail.innerHTML = `
    <div class="detail-row">
      <span class="label">Volume</span>
      <span class="value">${formatted.ml} ml</span>
    </div>
    <div class="detail-row">
      <span class="label">Cups</span>
      <span class="value">${formatted.cups} cups</span>
    </div>
    <div class="detail-row">
      <span class="label">Tablespoons</span>
      <span class="value">${formatted.tbsp} tbsp</span>
    </div>
    <div class="detail-row">
      <span class="label">Fluid Oz</span>
      <span class="value">${formatted.floz} fl oz</span>
    </div>
    <div class="detail-row">
      <span class="label">Est. Calories</span>
      <span class="value">~${cal} kcal</span>
    </div>
    <div class="detail-row">
      <span class="label">Proportion</span>
      <span class="value">${seg.percentage}%</span>
    </div>
  `;
}

function initViewerControls() {
  const btnWireframe = $('btn-wireframe');
  const btnSegments = $('btn-segments');
  const btnReset = $('btn-reset');
  const btnBack = $('btn-back');
  
  btnWireframe.addEventListener('click', () => {
    state.showWireframe = !state.showWireframe;
    btnWireframe.classList.toggle('active', state.showWireframe);
    state.sceneManager.toggleWireframe(state.showWireframe);
  });
  
  btnSegments.addEventListener('click', () => {
    state.showSegments = !state.showSegments;
    btnSegments.classList.toggle('active', state.showSegments);
    state.sceneManager.toggleSegments(state.showSegments);
  });
  
  btnReset.addEventListener('click', () => {
    state.sceneManager.resetView();
    selectSegment(null);
  });
  
  btnBack.addEventListener('click', () => {
    // Reset and go back
    state.images = [];
    state.segments = [];
    state.volumes = {};
    state.foodGroup = null;
    $('thumbnails').innerHTML = '';
    $('thumbnails').classList.add('hidden');
    $('scale-section').classList.add('hidden');
    $('process-btn').classList.add('hidden');
    $('file-input').value = '';
    
    // Reset processing steps
    $('processing-steps').querySelectorAll('.step').forEach(s => {
      s.classList.remove('active', 'done');
    });
    $('progress-bar').style.width = '0%';
    
    showScreen('upload');
  });
}

// ---- Tuning Panel ----
function initTuningPanel() {
  const tuningControls = $('tuning-controls');

  // ── Backend mode: replace browser controls with MapAnything info panel ──
  if (state.backendGlbLoaded) {
    tuningControls.innerHTML = `
      <div style="padding:8px 0 16px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <span style="font-size:20px;">🧠</span>
          <div>
            <div style="font-weight:600;color:#10b981;font-size:13px;">MapAnything Active</div>
            <div style="font-size:11px;color:var(--text-muted);">Metric 3D reconstruction on MPS</div>
          </div>
        </div>
        <div style="font-size:12px;color:var(--text-muted);line-height:1.5;margin-bottom:16px;">
          Depth estimation, mesh generation, and segmentation are handled
          by the local MapAnything model — browser controls are not used.
        </div>
        <div class="tune-group">
          <label>Plate diameter</label>
          <div style="display:flex;align-items:center;gap:8px;">
            <input type="number" id="backend-plate-diameter"
              value="${state.plateDiameter}" min="10" max="50" step="0.5"
              style="width:70px;padding:4px 8px;background:var(--surface);border:1px solid var(--border);
                     border-radius:6px;color:var(--text-primary);font-size:13px;" />
            <span style="font-size:12px;color:var(--text-muted);">cm</span>
          </div>
        </div>
        <button id="tune-apply" class="tune-apply-btn" style="margin-top:8px;">
          🔄 Re-run Reconstruction
        </button>
      </div>
    `;

    $('backend-plate-diameter').addEventListener('change', (e) => {
      state.plateDiameter = parseFloat(e.target.value) || 26;
    });

    $('tune-apply').addEventListener('click', async () => {
      const btn = $('tune-apply');
      btn.textContent = 'Processing...';
      btn.disabled = true;
      state.backendGlbLoaded = false;
      await startProcessing();
      btn.textContent = '🔄 Re-run Reconstruction';
      btn.disabled = false;
    });

    return; // Don't initialise the browser controls
  }

  // ── Browser-only mode: normal tuning controls ───────────────────────────
  const applyBtn = $('tune-apply');
  
  // Live value display for sliders
  const sliders = [
    { id: 'tune-cutoff', valId: 'val-cutoff', fixed: 2 },
    { id: 'tune-depthscale', valId: 'val-depthscale', fixed: 2 },
    { id: 'tune-smoothing', valId: 'val-smoothing', fixed: 0 },
    { id: 'tune-taper', valId: 'val-taper', fixed: 0 },
  ];
  
  sliders.forEach(({ id, valId, fixed }) => {
    const slider = $(id);
    const valSpan = $(valId);
    slider.addEventListener('input', () => {
      valSpan.textContent = parseFloat(slider.value).toFixed(fixed);
    });
  });
  
  // Regenerate on button click
  applyBtn.addEventListener('click', () => {
    regenerateModel();
  });
}


async function regenerateModel() {
  if (!state.cachedImageData || !state.cachedDepthResult) return;
  
  const applyBtn = $('tune-apply');
  applyBtn.textContent = 'Regenerating...';
  applyBtn.classList.add('regenerating');
  
  await sleep(50); // let UI update
  
  try {
    const maskCutoff = parseFloat($('tune-cutoff').value);
    const depthScaleFactor = parseFloat($('tune-depthscale').value);
    const smoothingPasses = parseInt($('tune-smoothing').value);
    const edgeTaper = parseInt($('tune-taper').value);
    const showWalls = $('tune-walls').checked;
    const showBottom = $('tune-bottom').checked;
    const depthModel = $('tune-depth-model').value;
    const meshMethod = $('tune-mesh-method').value;
    const maskMethod = $('tune-mask-method').value;
    
    const imageData = state.cachedImageData;
    let depthResult = state.cachedDepthResult;
    
    // Re-run depth estimation if model changed
    if (depthModel !== state.cachedModelName) {
      applyBtn.textContent = 'Loading new model...';
      depthResult = await runDepthEstimation(imageData, () => {}, depthModel);
      state.cachedDepthResult = depthResult;
      state.cachedModelName = depthModel;
    }
    
    // Run CLIPSeg if needed and not cached
    if (maskMethod === 'clipseg' && !state.cachedClipSegResult) {
      applyBtn.textContent = 'Running food detection...';
      state.cachedClipSegResult = await runClipSegWorker(imageData);
    }
    
    const autoLevel = $('tune-autolevel') && $('tune-autolevel').checked;
    
    // Re-run segmentation with new cutoff and mask method
    const segResult = segmentIngredients(
      imageData.data,
      imageData.width,
      imageData.height,
      depthResult.depth,
      depthResult.width,
      depthResult.height,
      6,
      maskCutoff,
      maskMethod,
      maskMethod === 'clipseg' ? state.cachedClipSegResult : null
    );
    state.segments = segResult.segments;
    
    // Re-generate mesh with new params
    const foodGroup = generateMesh(segResult, imageData.data, state.plateDiameter, {
      depthScaleFactor,
      smoothingPasses,
      edgeTaper,
      showWalls,
      showBottom,
      meshMethod,
    });
    state.foodGroup = foodGroup;
    
    // Re-calculate volumes
    const volumes = calculateVolumes(foodGroup);
    state.volumes = volumes;
    
    // Update 3D scene
    state.sceneManager.setFoodModel(state.foodGroup);
    state.sceneManager.onSegmentClick = (segId) => selectSegment(segId);
    
    // Update sidebar
    populateSidebar();
    
  } catch (error) {
    console.error('Regeneration error:', error);
  }
  
  applyBtn.textContent = 'Regenerate Model';
  applyBtn.classList.remove('regenerating');
}

// ---- Utility ----
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', async () => {
  initUpload();

  // Check if local MapAnything backend is running
  const backendStatus = await checkBackend();

  // Inject backend status badge into header
  const header = document.getElementById('app-header');
  if (header) {
    const badge = document.createElement('div');
    badge.id = 'backend-badge';
    badge.style.cssText = `
      display:flex;align-items:center;gap:6px;padding:4px 12px;
      border-radius:20px;font-size:12px;font-weight:600;
      ${ backendStatus.available
          ? 'background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);'
          : 'background:rgba(99,102,241,0.15);color:#6366f1;border:1px solid rgba(99,102,241,0.3);' }
    `;
    badge.innerHTML = backendStatus.available
      ? `<span>🟢</span><span>MapAnything AI Active (${backendStatus.device?.toUpperCase()})</span>`
      : `<span>🔵</span><span>Browser AI Mode</span>`;
    header.appendChild(badge);
  }
});
