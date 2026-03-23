/**
 * Food Ingredient Segmentation
 * K-means color clustering + depth edge detection for ingredient separation
 */

/**
 * Estimate plate tilt from the depth map and return corrected depth values.
 * Fits a plane to the non-food (background) depth values and subtracts it,
 * so depth represents height above the plate surface rather than distance to camera.
 */
export function correctPlateOrientation(depthData, foodMask, width, height) {
  // Collect background (non-food) depth samples for plane fitting
  const samples = [];
  const step = Math.max(1, Math.floor(width * height / 2000)); // max ~2000 samples
  for (let i = 0; i < width * height; i += step) {
    if (!foodMask[i] && isFinite(depthData[i]) && depthData[i] > 0) {
      const x = i % width;
      const y = Math.floor(i / width);
      samples.push({ x, y, z: depthData[i] });
    }
  }
  
  if (samples.length < 10) {
    // Not enough background samples, return copy unchanged
    return new Float32Array(depthData);
  }
  
  // Fit a plane z = ax + by + c using least squares
  let sumX = 0, sumY = 0, sumZ = 0;
  let sumXX = 0, sumXY = 0, sumYY = 0;
  let sumXZ = 0, sumYZ = 0;
  const n = samples.length;
  
  for (const s of samples) {
    sumX += s.x; sumY += s.y; sumZ += s.z;
    sumXX += s.x * s.x; sumXY += s.x * s.y; sumYY += s.y * s.y;
    sumXZ += s.x * s.z; sumYZ += s.y * s.z;
  }
  
  // Solve 3x3 system for [a, b, c]
  const A = [
    [sumXX, sumXY, sumX],
    [sumXY, sumYY, sumY],
    [sumX,  sumY,  n   ],
  ];
  const B = [sumXZ, sumYZ, sumZ];
  
  // Cramer's rule for 3x3
  const det = (m) => 
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  
  const D = det(A);
  if (Math.abs(D) < 1e-10) return new Float32Array(depthData);
  
  const Dx = det([[B[0], A[0][1], A[0][2]], [B[1], A[1][1], A[1][2]], [B[2], A[2][1], A[2][2]]]);
  const Dy = det([[A[0][0], B[0], A[0][2]], [A[1][0], B[1], A[1][2]], [A[2][0], B[2], A[2][2]]]);
  const Dc = det([[A[0][0], A[0][1], B[0]], [A[1][0], A[1][1], B[1]], [A[2][0], A[2][1], B[2]]]);
  
  const a = Dx / D;
  const b = Dy / D;
  const c = Dc / D;
  
  // Subtract the fitted plane from all depth values
  // This makes food height relative to the plate surface
  const corrected = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const x = i % width;
    const y = Math.floor(i / width);
    const planeZ = a * x + b * y + c;
    corrected[i] = depthData[i] - planeZ;
  }
  
  return corrected;
}

// Predefined food color names with heuristic labels
const FOOD_COLOR_LABELS = [
  { range: [0, 30], label: 'Red/Tomato' },
  { range: [30, 55], label: 'Orange/Carrot' },
  { range: [55, 75], label: 'Yellow/Corn' },
  { range: [75, 160], label: 'Green/Vegetable' },
  { range: [160, 200], label: 'Cyan/Herb' },
  { range: [200, 260], label: 'Blue/Berry' },
  { range: [260, 320], label: 'Purple/Eggplant' },
  { range: [320, 360], label: 'Pink/Salmon' },
];

const NEUTRAL_LABELS = [
  { maxSat: 15, minLight: 80, label: 'White/Rice' },
  { maxSat: 15, maxLight: 30, label: 'Dark/Sauce' },
  { maxSat: 25, label: 'Brown/Meat' },
];

/**
 * Convert RGB to HSL
 */
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;

  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }
  return [h * 360, s * 100, l * 100];
}

/**
 * K-means clustering in LAB-like color space
 */
function kMeansClustering(pixels, k, maxIter = 20) {
  const n = pixels.length;
  if (n === 0) return { centroids: [], labels: new Int32Array(0) };

  // Initialize centroids using k-means++ 
  const centroids = [];
  const randIdx = Math.floor(Math.random() * n);
  centroids.push([...pixels[randIdx]]);

  for (let c = 1; c < k; c++) {
    const distances = new Float32Array(n);
    let totalDist = 0;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const d = colorDistance(pixels[i], centroid);
        if (d < minDist) minDist = d;
      }
      distances[i] = minDist;
      totalDist += minDist;
    }
    // Weighted random selection
    let r = Math.random() * totalDist;
    for (let i = 0; i < n; i++) {
      r -= distances[i];
      if (r <= 0) {
        centroids.push([...pixels[i]]);
        break;
      }
    }
    if (centroids.length <= c) {
      centroids.push([...pixels[Math.floor(Math.random() * n)]]);
    }
  }

  const labels = new Int32Array(n);
  
  for (let iter = 0; iter < maxIter; iter++) {
    // Assignment step
    let changed = false;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const d = colorDistance(pixels[i], centroids[c]);
        if (d < minDist) {
          minDist = d;
          bestCluster = c;
        }
      }
      if (labels[i] !== bestCluster) {
        labels[i] = bestCluster;
        changed = true;
      }
    }
    if (!changed) break;

    // Update step
    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      sums[c][0] += pixels[i][0];
      sums[c][1] += pixels[i][1];
      sums[c][2] += pixels[i][2];
      counts[c]++;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        centroids[c][0] = sums[c][0] / counts[c];
        centroids[c][1] = sums[c][1] / counts[c];
        centroids[c][2] = sums[c][2] / counts[c];
      }
    }
  }

  return { centroids, labels };
}

function colorDistance(a, b) {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return dr * dr + dg * dg + db * db;
}

/**
 * Detect food region by finding the raised object above the dominant background surface.
 * Strategy: the most common depth value is the table/background.  Only keep pixels
 * whose depth is clearly above that level.
 */
function detectFoodRegion(imageData, depthData, width, height, depthWidth, depthHeight, maskCutoff = 0.15, maskMethod = 'histogram') {
  const mask = new Uint8Array(width * height);
  
  // Resize depth to match image dimensions
  const resizedDepth = resizeDepthMap(depthData, depthWidth, depthHeight, width, height);
  
  if (maskMethod === 'otsu') {
    // ---- OTSU THRESHOLD METHOD ----
    const numBins = 256;
    const histogram = new Float32Array(numBins);
    for (let i = 0; i < resizedDepth.length; i++) {
      const bin = Math.min(numBins - 1, Math.floor(resizedDepth[i] * (numBins - 1)));
      histogram[bin]++;
    }
    let totalSum = 0, totalCount = resizedDepth.length;
    for (let i = 0; i < numBins; i++) totalSum += i * histogram[i];
    let sumB = 0, wB = 0, maxVariance = 0, threshold = 128;
    for (let t = 0; t < numBins; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      const wF = totalCount - wB;
      if (wF === 0) break;
      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (totalSum - sumB) / wF;
      const variance = wB * wF * (mB - mF) * (mB - mF);
      if (variance > maxVariance) { maxVariance = variance; threshold = t; }
    }
    const depthThreshold = (threshold / (numBins - 1)) * (1 + maskCutoff);
    for (let i = 0; i < resizedDepth.length; i++) {
      mask[i] = resizedDepth[i] >= depthThreshold ? 1 : 0;
    }
  } else if (maskMethod === 'fixed') {
    // ---- FIXED DEPTH CUTOFF ----
    // maskCutoff directly controls the threshold (0 = everything, 0.5 = top half only)
    const cutoff = maskCutoff;
    for (let i = 0; i < resizedDepth.length; i++) {
      mask[i] = resizedDepth[i] > cutoff ? 1 : 0;
    }
  } else {
    // ---- HISTOGRAM MODE METHOD (default) ----
    const numBins = 256;
    const histogram = new Float32Array(numBins);
    for (let i = 0; i < resizedDepth.length; i++) {
      const bin = Math.min(numBins - 1, Math.floor(resizedDepth[i] * (numBins - 1)));
      histogram[bin]++;
    }
    const smoothed = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) {
      let sum = 0, cnt = 0;
      for (let j = Math.max(0, i - 3); j <= Math.min(numBins - 1, i + 3); j++) {
        sum += histogram[j]; cnt++;
      }
      smoothed[i] = sum / cnt;
    }
    let peakBin = 0, peakVal = 0;
    for (let i = 0; i < numBins; i++) {
      if (smoothed[i] > peakVal) { peakVal = smoothed[i]; peakBin = i; }
    }
    const backgroundDepth = peakBin / (numBins - 1);
    let maxDepth = 0;
    for (let i = 0; i < resizedDepth.length; i++) {
      if (resizedDepth[i] > maxDepth) maxDepth = resizedDepth[i];
    }
    const depthRange = maxDepth - backgroundDepth;
    const margin = depthRange * maskCutoff;
    const cutoff = backgroundDepth + margin;
    for (let i = 0; i < resizedDepth.length; i++) {
      mask[i] = resizedDepth[i] > cutoff ? 1 : 0;
    }
  }
  
  // Morphological cleanup
  morphErode(mask, width, height, 3);
  morphDilate(mask, width, height, 3);
  
  // Keep only the largest connected component
  keepLargestComponent(mask, width, height);
  
  return { mask, resizedDepth };
}

/**
 * Keep only the largest connected component in a binary mask (flood-fill based)
 */
function keepLargestComponent(mask, width, height) {
  const labels = new Int32Array(width * height);
  let nextLabel = 1;
  const componentSizes = new Map();
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 0 || labels[idx] !== 0) continue;
      
      // BFS flood fill
      const label = nextLabel++;
      const queue = [idx];
      let size = 0;
      labels[idx] = label;
      
      while (queue.length > 0) {
        const cur = queue.pop();
        size++;
        const cx = cur % width, cy = Math.floor(cur / width);
        
        for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const ni = ny * width + nx;
          if (mask[ni] === 1 && labels[ni] === 0) {
            labels[ni] = label;
            queue.push(ni);
          }
        }
      }
      componentSizes.set(label, size);
    }
  }
  
  // Find the largest component
  let largestLabel = 0, largestSize = 0;
  for (const [label, size] of componentSizes) {
    if (size > largestSize) { largestSize = size; largestLabel = label; }
  }
  
  // Zero out everything except the largest component
  for (let i = 0; i < mask.length; i++) {
    mask[i] = labels[i] === largestLabel ? 1 : 0;
  }
}

/**
 * Morphological erosion (shrink mask by radius pixels)
 */
function morphErode(mask, width, height, radius) {
  const temp = new Uint8Array(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 0) { temp[idx] = 0; continue; }
      let keep = true;
      outer: for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height || mask[ny * width + nx] === 0) {
            keep = false; break outer;
          }
        }
      }
      temp[idx] = keep ? 1 : 0;
    }
  }
  for (let i = 0; i < mask.length; i++) mask[i] = temp[i];
}

/**
 * Morphological dilation (grow mask by radius pixels)
 */
function morphDilate(mask, width, height, radius) {
  const temp = new Uint8Array(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 1) { temp[idx] = 1; continue; }
      let grow = false;
      outer: for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height && mask[ny * width + nx] === 1) {
            grow = true; break outer;
          }
        }
      }
      temp[idx] = grow ? 1 : 0;
    }
  }
  for (let i = 0; i < mask.length; i++) mask[i] = temp[i];
}

/**
 * Bilinear interpolation resize for depth map
 */
function resizeDepthMap(depthData, srcW, srcH, dstW, dstH) {
  const result = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const srcX = x * xRatio;
      const srcY = y * yRatio;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const y1 = Math.min(y0 + 1, srcH - 1);
      const xFrac = srcX - x0;
      const yFrac = srcY - y0;
      
      const v00 = depthData[y0 * srcW + x0];
      const v10 = depthData[y0 * srcW + x1];
      const v01 = depthData[y1 * srcW + x0];
      const v11 = depthData[y1 * srcW + x1];
      
      result[y * dstW + x] = 
        v00 * (1 - xFrac) * (1 - yFrac) +
        v10 * xFrac * (1 - yFrac) +
        v01 * (1 - xFrac) * yFrac +
        v11 * xFrac * yFrac;
    }
  }
  return result;
}

/**
 * Label food ingredient name based on centroid color
 */
function labelIngredient(centroid) {
  const [h, s, l] = rgbToHsl(centroid[0], centroid[1], centroid[2]);
  
  // Check neutrals first
  if (s < 15 && l > 75) return 'White/Rice';
  if (s < 15 && l < 25) return 'Dark/Sauce';
  if (s < 25 && l > 20 && l < 60) return 'Brown/Meat';
  if (s < 25) return 'Beige/Bread';
  
  // Check by hue
  for (const entry of FOOD_COLOR_LABELS) {
    if (h >= entry.range[0] && h < entry.range[1]) return entry.label;
  }
  
  return 'Other';
}

/**
 * Generate display-friendly color for a segment
 */
function centroidToHex(centroid) {
  const r = Math.round(Math.min(255, Math.max(0, centroid[0])));
  const g = Math.round(Math.min(255, Math.max(0, centroid[1])));
  const b = Math.round(Math.min(255, Math.max(0, centroid[2])));
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

// Segment highlight colors (vibrant palette for 3D view)
const SEGMENT_COLORS = [
  '#6366f1', '#a855f7', '#ec4899', '#f43f5e',
  '#f59e0b', '#22c55e', '#06b6d4', '#3b82f6',
  '#8b5cf6', '#14b8a6', '#f97316', '#84cc16',
];

/**
 * Main segmentation function
 * @returns {object} segments with labels, colors, masks, pixel counts
 */
export function segmentIngredients(imageData, width, height, depthData, depthWidth, depthHeight, numSegments = 6, maskCutoff = 0.15, maskMethod = 'histogram', externalMask = null) {
  let foodMask, resizedDepth;
  
  if (maskMethod === 'clipseg' && externalMask) {
    // Use the externally provided CLIPSeg mask
    // Resize CLIPSeg mask to match image dimensions
    const { mask: clipMask, maskWidth, maskHeight } = externalMask;
    foodMask = new Uint8Array(width * height);
    const xRatio = maskWidth / width;
    const yRatio = maskHeight / height;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const sx = Math.min(Math.floor(x * xRatio), maskWidth - 1);
        const sy = Math.min(Math.floor(y * yRatio), maskHeight - 1);
        const prob = clipMask[sy * maskWidth + sx];
        foodMask[y * width + x] = prob > 0.5 ? 1 : 0; // threshold at 0.5
      }
    }
    // Resize depth to match image
    resizedDepth = new Float32Array(width * height);
    const dxRatio = depthWidth / width;
    const dyRatio = depthHeight / height;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const sx = Math.min(Math.floor(x * dxRatio), depthWidth - 1);
        const sy = Math.min(Math.floor(y * dyRatio), depthHeight - 1);
        resizedDepth[y * width + x] = depthData[sy * depthWidth + sx];
      }
    }
  } else {
    // Use depth-based food region detection
    const result = detectFoodRegion(
      imageData, depthData, width, height, depthWidth, depthHeight, maskCutoff, maskMethod
    );
    foodMask = result.mask;
    resizedDepth = result.resizedDepth;
  }
  
  // 2. Collect food pixels for clustering
  const foodPixels = [];
  const foodIndices = [];
  
  for (let i = 0; i < width * height; i++) {
    if (foodMask[i]) {
      const r = imageData[i * 4];
      const g = imageData[i * 4 + 1];
      const b = imageData[i * 4 + 2];
      foodPixels.push([r, g, b]);
      foodIndices.push(i);
    }
  }
  
  if (foodPixels.length === 0) {
    // Fallback: use all pixels
    for (let i = 0; i < width * height; i++) {
      const r = imageData[i * 4];
      const g = imageData[i * 4 + 1];
      const b = imageData[i * 4 + 2];
      foodPixels.push([r, g, b]);
      foodIndices.push(i);
      foodMask[i] = 1;
    }
  }
  
  // 3. Subsample for faster clustering
  const maxSamples = 50000;
  let samplePixels = foodPixels;
  let sampleIndices;
  
  if (foodPixels.length > maxSamples) {
    const step = Math.ceil(foodPixels.length / maxSamples);
    samplePixels = [];
    sampleIndices = [];
    for (let i = 0; i < foodPixels.length; i += step) {
      samplePixels.push(foodPixels[i]);
      sampleIndices.push(i);
    }
  }
  
  // Determine optimal k based on color variance
  const actualK = Math.min(numSegments, Math.max(2, Math.ceil(foodPixels.length / 5000)));
  
  // 4. K-means clustering on sampled pixels
  const { centroids, labels: sampleLabels } = kMeansClustering(samplePixels, actualK);
  
  // 5. Assign all food pixels to nearest centroid
  const pixelLabels = new Int32Array(width * height).fill(-1);
  
  for (let i = 0; i < foodPixels.length; i++) {
    let minDist = Infinity;
    let bestC = 0;
    for (let c = 0; c < centroids.length; c++) {
      const d = colorDistance(foodPixels[i], centroids[c]);
      if (d < minDist) {
        minDist = d;
        bestC = c;
      }
    }
    pixelLabels[foodIndices[i]] = bestC;
  }
  
  // 6. Build segment info
  const segments = centroids.map((centroid, idx) => {
    let count = 0;
    for (let i = 0; i < pixelLabels.length; i++) {
      if (pixelLabels[i] === idx) count++;
    }
    
    return {
      id: idx,
      label: labelIngredient(centroid),
      color: SEGMENT_COLORS[idx % SEGMENT_COLORS.length],
      originalColor: centroidToHex(centroid),
      centroid: centroid,
      pixelCount: count,
      percentage: 0, // calculated after
    };
  });
  
  // Filter out very small segments (< 2% of food area)
  const totalFoodPixels = foodIndices.length;
  const filtered = segments.filter(s => s.pixelCount / totalFoodPixels > 0.02);
  
  // Recalculate percentages
  const filteredTotal = filtered.reduce((sum, s) => sum + s.pixelCount, 0);
  filtered.forEach(s => {
    s.percentage = Math.round((s.pixelCount / filteredTotal) * 100);
  });
  
  // Deduplicate labels
  const labelCounts = {};
  filtered.forEach(s => {
    labelCounts[s.label] = (labelCounts[s.label] || 0) + 1;
  });
  const labelIndexes = {};
  filtered.forEach(s => {
    if (labelCounts[s.label] > 1) {
      labelIndexes[s.label] = (labelIndexes[s.label] || 0) + 1;
      s.label = `${s.label} ${labelIndexes[s.label]}`;
    }
  });
  
  return {
    segments: filtered,
    pixelLabels,
    foodMask,
    resizedDepth,
    width,
    height
  };
}
