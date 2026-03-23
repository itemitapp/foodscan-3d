/**
 * CLIPSeg Segmentation Web Worker
 * Uses CLIPSeg for text-prompted food segmentation via Transformers.js
 */

import { AutoProcessor, CLIPSegForImageSegmentation, RawImage } from '@huggingface/transformers';

let processor = null;
let model = null;

async function loadModel() {
  if (model && processor) return;
  
  self.postMessage({ type: 'status', message: 'Loading CLIPSeg food segmentation model...' });
  
  const modelId = 'Xenova/clipseg-rd64-refined';
  
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
  
  processor = await AutoProcessor.from_pretrained(modelId, { progress_callback: progressCb });
  model = await CLIPSegForImageSegmentation.from_pretrained(modelId, {
    device: 'wasm',
    dtype: 'fp32',
    progress_callback: progressCb,
  });
  
  self.postMessage({ type: 'status', message: 'CLIPSeg model loaded' });
}

async function segment(imageBuffer, width, height, prompts) {
  await loadModel();
  
  self.postMessage({ type: 'status', message: 'Running food segmentation...' });
  
  // Create RawImage from RGBA buffer
  const uint8Data = new Uint8Array(imageBuffer);
  // Convert RGBA to RGB
  const rgbData = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgbData[i * 3] = uint8Data[i * 4];
    rgbData[i * 3 + 1] = uint8Data[i * 4 + 1];
    rgbData[i * 3 + 2] = uint8Data[i * 4 + 2];
  }
  
  const image = new RawImage(rgbData, width, height, 3);
  
  // Run CLIPSeg with food-related prompts
  const inputs = await processor(image, prompts, { padding: true, return_tensors: 'pt' });
  const outputs = await model(inputs);
  
  // Get logits and convert to probabilities
  const logits = outputs.logits; // shape: [num_prompts, H, W]
  const logitsData = logits.data;
  const maskH = logits.dims[logits.dims.length - 2];
  const maskW = logits.dims[logits.dims.length - 1];
  const numPrompts = prompts.length;
  const maskSize = maskH * maskW;
  
  // Combine multiple prompt masks by taking the max probability
  const combinedMask = new Float32Array(maskSize);
  for (let i = 0; i < maskSize; i++) {
    let maxVal = -Infinity;
    for (let p = 0; p < numPrompts; p++) {
      const val = logitsData[p * maskSize + i];
      if (val > maxVal) maxVal = val;
    }
    // Sigmoid to convert logits to probability
    combinedMask[i] = 1 / (1 + Math.exp(-maxVal));
  }
  
  self.postMessage({
    type: 'result',
    mask: combinedMask.buffer,
    maskWidth: maskW,
    maskHeight: maskH,
  }, [combinedMask.buffer]);
}

// Handle messages
self.onmessage = (e) => {
  const { type } = e.data;
  
  if (type === 'segment') {
    const { imageData, width, height, prompts } = e.data;
    segment(imageData, width, height, prompts || ['food', 'plate of food', 'meal'])
      .catch(err => {
        self.postMessage({ type: 'error', message: err.message });
      });
  }
};
