/**
 * Depth Estimation Web Worker
 * Uses Depth Anything V2 Small via Transformers.js
 */

import { pipeline, env, RawImage } from '@huggingface/transformers';

// Configure Transformers.js for browser
env.allowLocalModels = false;

let depthEstimators = {}; // cache by model name
let currentModelName = null;

async function loadModel(modelName = 'depth-anything-small') {
  const fullName = `onnx-community/${modelName}`;
  if (depthEstimators[fullName]) {
    currentModelName = fullName;
    return depthEstimators[fullName];
  }

  self.postMessage({ type: 'status', message: `Loading ${modelName} model...` });

  const progressCb = (progress) => {
    if (progress.status === 'progress') {
      self.postMessage({
        type: 'model-progress',
        loaded: progress.loaded,
        total: progress.total,
        file: progress.file,
      });
    }
  };

  let estimator;
  try {
    estimator = await pipeline('depth-estimation', fullName, {
      device: 'webgpu',
      dtype: 'fp32',
      progress_callback: progressCb,
    });
  } catch {
    self.postMessage({ type: 'status', message: 'WebGPU unavailable, using WASM fallback...' });
    estimator = await pipeline('depth-estimation', fullName, {
      device: 'wasm',
      dtype: 'fp32',
      progress_callback: progressCb,
    });
  }

  depthEstimators[fullName] = estimator;
  currentModelName = fullName;
  return estimator;
}

async function estimateDepth(imageBuffer, width, height, modelName) {
  const estimator = await loadModel(modelName);

  self.postMessage({ type: 'status', message: 'Running depth estimation...' });

  // Build a RawImage directly from the raw RGBA pixel buffer
  const rgba = new Uint8ClampedArray(imageBuffer);
  const rawImg = new RawImage(rgba, width, height, 4);

  const result = await estimator(rawImg);

  // Transformers.js returns { predicted_depth: Tensor, depth: RawImage }
  // predicted_depth is a Tensor with .data (Float32Array) and .dims [1, H, W]
  // depth is a RawImage normalized to 0-255

  let depthData, depthWidth, depthHeight;

  if (result.predicted_depth) {
    const tensor = result.predicted_depth;
    depthData = tensor.data;                   // Float32Array
    if (tensor.dims && tensor.dims.length === 3) {
      depthHeight = tensor.dims[1];
      depthWidth  = tensor.dims[2];
    } else if (tensor.dims && tensor.dims.length === 2) {
      depthHeight = tensor.dims[0];
      depthWidth  = tensor.dims[1];
    } else {
      // Fallback: assume same aspect as input
      depthWidth  = width;
      depthHeight = height;
    }
  } else if (result.depth) {
    // Fallback to the normalised RawImage
    const img = result.depth;
    depthData   = new Float32Array(img.data.length);
    for (let i = 0; i < img.data.length; i++) depthData[i] = img.data[i] / 255;
    depthWidth  = img.width;
    depthHeight = img.height;
  } else {
    throw new Error('Unexpected depth estimation output format');
  }

  // Normalise to 0-1 range
  let minD = Infinity, maxD = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    const v = depthData[i];
    if (!isFinite(v)) continue;
    if (v < minD) minD = v;
    if (v > maxD) maxD = v;
  }
  const range = maxD - minD || 1;
  const normalised = new Float32Array(depthWidth * depthHeight);
  const expectedLen = depthWidth * depthHeight;
  for (let i = 0; i < expectedLen; i++) {
    const v = i < depthData.length ? depthData[i] : 0;
    normalised[i] = isFinite(v) ? (v - minD) / range : 0;
  }

  return { depth: normalised, width: depthWidth, height: depthHeight, minDepth: minD, maxDepth: maxD };
}

// Listen for messages
self.onmessage = async (e) => {
  const { type, imageData, width, height, modelName } = e.data;

  if (type === 'estimate') {
    try {
      const result = await estimateDepth(imageData, width, height, modelName);
      self.postMessage(
        {
          type: 'result',
          depth: result.depth,
          width: result.width,
          height: result.height,
          minDepth: result.minDepth,
          maxDepth: result.maxDepth,
        },
        [result.depth.buffer],
      );
    } catch (error) {
      self.postMessage({ type: 'error', message: error.message || String(error) });
    }
  }
};
