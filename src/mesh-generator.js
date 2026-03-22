/**
 * 3D Mesh Generator
 * Converts depth map + segmentation data into a textured 3D displacement mesh
 */

import * as THREE from 'three';

/**
 * Creates a 3D mesh from depth map data with segment coloring
 * @param {object} segmentationResult - from segmentation.js
 * @param {Uint8ClampedArray} originalImageData - RGBA pixel data
 * @param {number} plateDiameter - in cm, for scale reference
 * @returns {THREE.Group} group containing mesh and segment sub-meshes
 */
export function generateMesh(segmentationResult, originalImageData, plateDiameter = 26, params = {}) {
  const {
    depthScaleFactor = 0.3,
    smoothingPasses = 2,
    edgeTaper: edgeTaperRadius = 5,
    showWalls = true,
    showBottom = true,
    meshMethod = 'displacement',
  } = params;
  
  // If point cloud requested, delegate to that function
  if (meshMethod === 'pointcloud') {
    return generatePointCloud(segmentationResult, originalImageData, plateDiameter, params);
  }
  
  const { segments, pixelLabels, foodMask, resizedDepth, width, height } = segmentationResult;
  
  // Determine mesh resolution (downsample for performance)
  const maxRes = 256;
  const scale = Math.min(maxRes / width, maxRes / height, 1);
  const meshW = Math.floor(width * scale);
  const meshH = Math.floor(height * scale);
  
  // Resample data to mesh resolution
  const meshDepth = resampleFloat32(resizedDepth, width, height, meshW, meshH);
  const meshLabels = resampleInt32(pixelLabels, width, height, meshW, meshH);
  const meshMask = resampleUint8(foodMask, width, height, meshW, meshH);
  const meshColors = resampleRGBA(originalImageData, width, height, meshW, meshH);
  
  // Apply 5x5 box blur to smooth depth (stronger smoothing for cleaner surfaces)
  for (let pass = 0; pass < smoothingPasses; pass++) {
    const smoothedDepth = new Float32Array(meshW * meshH);
    for (let y = 0; y < meshH; y++) {
      for (let x = 0; x < meshW; x++) {
        let sum = 0, count = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < meshW && ny >= 0 && ny < meshH) {
              const v = meshDepth[ny * meshW + nx];
              if (isFinite(v)) { sum += v; count++; }
            }
          }
        }
        smoothedDepth[y * meshW + x] = count > 0 ? sum / count : 0;
      }
    }
    for (let i = 0; i < smoothedDepth.length; i++) meshDepth[i] = smoothedDepth[i];
  }

  // Smooth depth at food mask edges to avoid hanging walls
  const edgeTaper = edgeTaperRadius;
  const taperWeights = new Float32Array(meshW * meshH);
  for (let y = 0; y < meshH; y++) {
    for (let x = 0; x < meshW; x++) {
      const idx = y * meshW + x;
      if (meshMask[idx] === 0) { taperWeights[idx] = 0; continue; }
      let minDist = edgeTaper + 1;
      for (let dy = -edgeTaper; dy <= edgeTaper; dy++) {
        for (let dx = -edgeTaper; dx <= edgeTaper; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= meshW || ny < 0 || ny >= meshH) {
            minDist = Math.min(minDist, Math.sqrt(dx*dx + dy*dy));
          } else if (meshMask[ny * meshW + nx] === 0) {
            minDist = Math.min(minDist, Math.sqrt(dx*dx + dy*dy));
          }
        }
      }
      taperWeights[idx] = Math.min(minDist / edgeTaper, 1);
    }
  }
  
  // Scale: plate diameter maps to mesh width
  const worldScale = plateDiameter / meshW; // cm per pixel
  const depthScale = plateDiameter * depthScaleFactor;
  
  // Build geometry
  const group = new THREE.Group();
  
  // Create main geometry
  const geometry = new THREE.BufferGeometry();
  const vertices = [];
  const normals = [];
  const colors = [];
  const uvs = [];
  const indices = [];
  const segmentIds = []; // custom attribute for raycasting
  
  // Create vertex grid (top surface)
  const centerX = meshW / 2;
  const centerY = meshH / 2;
  
  for (let y = 0; y < meshH; y++) {
    for (let x = 0; x < meshW; x++) {
      const idx = y * meshW + x;
      const rawDepth = meshDepth[idx];
      const depth = isFinite(rawDepth) ? rawDepth : 0;
      const isFoodPixel = meshMask[idx] > 0;
      const taper = taperWeights[idx];
      
      // Position: x,z on plane, y is height from depth (tapered at edges)
      const px = (x - centerX) * worldScale;
      const pz = (y - centerY) * worldScale;
      const py = isFoodPixel ? depth * depthScale * taper : 0;
      
      // NaN guard
      vertices.push(
        isFinite(px) ? px : 0,
        isFinite(py) ? py : 0,
        isFinite(pz) ? pz : 0
      );
      
      // UV
      uvs.push(x / meshW, 1 - y / meshH);
      
      // Color from original image
      const ci = idx * 4;
      colors.push(
        meshColors[ci] / 255,
        meshColors[ci + 1] / 255,
        meshColors[ci + 2] / 255
      );
      
      // Segment ID
      segmentIds.push(meshLabels[idx]);
    }
  }
  
  // ---- TOP SURFACE: create faces for ALL food-masked quads (no height threshold) ----
  for (let y = 0; y < meshH - 1; y++) {
    for (let x = 0; x < meshW - 1; x++) {
      const a = y * meshW + x;
      const b = y * meshW + (x + 1);
      const c = (y + 1) * meshW + x;
      const d = (y + 1) * meshW + (x + 1);
      
      const maskA = meshMask[a], maskB = meshMask[b];
      const maskC = meshMask[c], maskD = meshMask[d];
      
      // Triangle 1: a-b-c
      if (maskA && maskB && maskC) {
        indices.push(a, b, c);
      }
      // Triangle 2: b-d-c
      if (maskB && maskD && maskC) {
        indices.push(b, d, c);
      }
    }
  }
  
  // ---- SIDE WALLS: connect food mask boundary edges to y=0 ----
  const bottomVertexMap = new Map();
  
  function getOrCreateBottomVertex(topIdx) {
    if (bottomVertexMap.has(topIdx)) return bottomVertexMap.get(topIdx);
    const bIdx = vertices.length / 3;
    vertices.push(vertices[topIdx * 3], 0, vertices[topIdx * 3 + 2]);
    colors.push(colors[topIdx * 3], colors[topIdx * 3 + 1], colors[topIdx * 3 + 2]);
    uvs.push(uvs[topIdx * 2], uvs[topIdx * 2 + 1]);
    segmentIds.push(segmentIds[topIdx]);
    bottomVertexMap.set(topIdx, bIdx);
    return bIdx;
  }
  
  // Walk the food mask boundary — only create walls at the outer perimeter
  if (showWalls) {
  for (let y = 0; y < meshH; y++) {
    for (let x = 0; x < meshW; x++) {
      const idx = y * meshW + x;
      if (!meshMask[idx]) continue;
      
      // Check 4 directions for boundary
      // Right boundary
      if (x + 1 >= meshW || !meshMask[y * meshW + (x + 1)]) {
        if (y + 1 < meshH && meshMask[(y + 1) * meshW + x]) {
          const v0 = y * meshW + x;
          const v1 = (y + 1) * meshW + x;
          const b0 = getOrCreateBottomVertex(v0);
          const b1 = getOrCreateBottomVertex(v1);
          indices.push(v0, v1, b1);
          indices.push(v0, b1, b0);
        }
      }
      // Left boundary
      if (x - 1 < 0 || !meshMask[y * meshW + (x - 1)]) {
        if (y + 1 < meshH && meshMask[(y + 1) * meshW + x]) {
          const v0 = (y + 1) * meshW + x;
          const v1 = y * meshW + x;
          const b0 = getOrCreateBottomVertex(v0);
          const b1 = getOrCreateBottomVertex(v1);
          indices.push(v0, v1, b1);
          indices.push(v0, b1, b0);
        }
      }
      // Bottom boundary
      if (y + 1 >= meshH || !meshMask[(y + 1) * meshW + x]) {
        if (x + 1 < meshW && meshMask[y * meshW + (x + 1)]) {
          const v0 = y * meshW + (x + 1);
          const v1 = y * meshW + x;
          const b0 = getOrCreateBottomVertex(v0);
          const b1 = getOrCreateBottomVertex(v1);
          indices.push(v0, v1, b1);
          indices.push(v0, b1, b0);
        }
      }
      // Top boundary
      if (y - 1 < 0 || !meshMask[(y - 1) * meshW + x]) {
        if (x + 1 < meshW && meshMask[y * meshW + (x + 1)]) {
          const v0 = y * meshW + x;
          const v1 = y * meshW + (x + 1);
          const b0 = getOrCreateBottomVertex(v0);
          const b1 = getOrCreateBottomVertex(v1);
          indices.push(v0, v1, b1);
          indices.push(v0, b1, b0);
        }
      }
    }
  }
  } // end showWalls
  
  // ---- BOTTOM CAP: flat surface at y=0 for food-masked cells ----
  if (showBottom) {
  for (let y = 0; y < meshH - 1; y++) {
    for (let x = 0; x < meshW - 1; x++) {
      const a = y * meshW + x;
      const b = y * meshW + (x + 1);
      const c = (y + 1) * meshW + x;
      const d = (y + 1) * meshW + (x + 1);
      
      if (meshMask[a] && meshMask[b] && meshMask[c]) {
        const ba = getOrCreateBottomVertex(a);
        const bb = getOrCreateBottomVertex(b);
        const bc = getOrCreateBottomVertex(c);
        indices.push(bc, bb, ba);
      }
      if (meshMask[b] && meshMask[d] && meshMask[c]) {
        const bb = getOrCreateBottomVertex(b);
        const bd = getOrCreateBottomVertex(d);
        const bc = getOrCreateBottomVertex(c);
        indices.push(bc, bd, bb);
      }
    }
  }
  } // end showBottom
  
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('segmentId', new THREE.Int32BufferAttribute(segmentIds, 1));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  
  // Main mesh with vertex colors
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.6,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });
  
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'foodMesh';
  mesh.userData = { segments, pixelLabels: meshLabels, meshW, meshH, worldScale, depthScale };
  group.add(mesh);
  
  // Create segment highlight overlay (initially invisible)
  const segColorArray = new Float32Array(vertices.length);
  for (let i = 0; i < segmentIds.length; i++) {
    const segId = segmentIds[i];
    const seg = segments.find(s => s.id === segId);
    if (seg) {
      const c = new THREE.Color(seg.color);
      segColorArray[i * 3] = c.r;
      segColorArray[i * 3 + 1] = c.g;
      segColorArray[i * 3 + 2] = c.b;
    } else {
      segColorArray[i * 3] = 0.15;
      segColorArray[i * 3 + 1] = 0.15;
      segColorArray[i * 3 + 2] = 0.15;
    }
  }
  
  const segGeometry = geometry.clone();
  segGeometry.setAttribute('color', new THREE.Float32BufferAttribute(segColorArray, 3));
  
  const segMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.5,
    metalness: 0.15,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.85,
  });
  
  const segMesh = new THREE.Mesh(segGeometry, segMaterial);
  segMesh.name = 'segmentMesh';
  segMesh.visible = true;
  segMesh.position.y = 0.02; // slight offset to prevent z-fighting
  group.add(segMesh);
  
  // Create wireframe
  const wireGeometry = new THREE.WireframeGeometry(geometry);
  const wireMaterial = new THREE.LineBasicMaterial({
    color: 0x6366f1,
    transparent: true,
    opacity: 0.15,
  });
  const wireframe = new THREE.LineSegments(wireGeometry, wireMaterial);
  wireframe.name = 'wireframe';
  wireframe.visible = false;
  group.add(wireframe);
  
  return group;
}

/**
 * Highlight a specific segment on the mesh
 */
export function highlightSegment(group, segmentId) {
  const segMesh = group.getObjectByName('segmentMesh');
  if (!segMesh) return;
  
  const foodMesh = group.getObjectByName('foodMesh');
  const segments = foodMesh.userData.segments;
  const segIds = foodMesh.geometry.getAttribute('segmentId').array;
  const colorAttr = segMesh.geometry.getAttribute('color');
  
  for (let i = 0; i < segIds.length; i++) {
    const sid = segIds[i];
    const seg = segments.find(s => s.id === sid);
    
    if (segmentId !== null && sid === segmentId) {
      // Brighten selected segment
      if (seg) {
        const c = new THREE.Color(seg.color).multiplyScalar(1.3);
        colorAttr.setXYZ(i, c.r, c.g, c.b);
      }
    } else if (segmentId !== null) {
      // Dim other segments
      if (seg) {
        const c = new THREE.Color(seg.color).multiplyScalar(0.3);
        colorAttr.setXYZ(i, c.r, c.g, c.b);
      } else {
        colorAttr.setXYZ(i, 0.05, 0.05, 0.05);
      }
    } else {
      // Reset all
      if (seg) {
        const c = new THREE.Color(seg.color);
        colorAttr.setXYZ(i, c.r, c.g, c.b);
      } else {
        colorAttr.setXYZ(i, 0.15, 0.15, 0.15);
      }
    }
  }
  
  colorAttr.needsUpdate = true;
}

/**
 * Get segment ID at a raycasted point
 */
export function getSegmentAtPoint(intersection, group) {
  const foodMesh = group.getObjectByName('foodMesh');
  if (!foodMesh || !intersection) return -1;
  
  const face = intersection.face;
  if (!face) return -1;
  
  const segIds = foodMesh.geometry.getAttribute('segmentId').array;
  return segIds[face.a];
}

/**
 * Generate a colored 3D point cloud (alternative to displacement mesh)
 */
function generatePointCloud(segmentationResult, originalImageData, plateDiameter, params = {}) {
  const { depthScaleFactor = 0.3, smoothingPasses = 0 } = params;
  const { segments, pixelLabels, foodMask, resizedDepth, width, height } = segmentationResult;
  
  const maxRes = 256;
  const scale = Math.min(maxRes / width, maxRes / height, 1);
  const meshW = Math.floor(width * scale);
  const meshH = Math.floor(height * scale);
  
  const meshDepth = resampleFloat32(resizedDepth, width, height, meshW, meshH);
  const meshLabels = resampleInt32(pixelLabels, width, height, meshW, meshH);
  const meshMask = resampleUint8(foodMask, width, height, meshW, meshH);
  const meshColors = resampleRGBA(originalImageData, width, height, meshW, meshH);
  
  // Optional smoothing
  for (let pass = 0; pass < smoothingPasses; pass++) {
    const smoothed = new Float32Array(meshW * meshH);
    for (let y = 0; y < meshH; y++) {
      for (let x = 0; x < meshW; x++) {
        let sum = 0, count = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < meshW && ny >= 0 && ny < meshH) {
              const v = meshDepth[ny * meshW + nx];
              if (isFinite(v)) { sum += v; count++; }
            }
          }
        }
        smoothed[y * meshW + x] = count > 0 ? sum / count : 0;
      }
    }
    for (let i = 0; i < smoothed.length; i++) meshDepth[i] = smoothed[i];
  }
  
  const worldScale = plateDiameter / meshW;
  const depthScale = plateDiameter * depthScaleFactor;
  const centerX = meshW / 2;
  const centerY = meshH / 2;
  
  const positions = [];
  const colors = [];
  const segmentIds = [];
  
  for (let y = 0; y < meshH; y++) {
    for (let x = 0; x < meshW; x++) {
      const idx = y * meshW + x;
      if (!meshMask[idx]) continue;
      
      const depth = isFinite(meshDepth[idx]) ? meshDepth[idx] : 0;
      positions.push(
        (x - centerX) * worldScale,
        depth * depthScale,
        (y - centerY) * worldScale
      );
      
      const ci = idx * 4;
      colors.push(meshColors[ci] / 255, meshColors[ci + 1] / 255, meshColors[ci + 2] / 255);
      segmentIds.push(meshLabels[idx]);
    }
  }
  
  const group = new THREE.Group();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('segmentId', new THREE.Int32BufferAttribute(segmentIds, 1));
  
  const material = new THREE.PointsMaterial({
    size: worldScale * 1.2,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  const points = new THREE.Points(geometry, material);
  points.name = 'foodMesh';
  points.userData = { segments, pixelLabels: meshLabels, meshW, meshH, worldScale, depthScale };
  group.add(points);
  
  // Segment overlay (also points)
  const segColorArray = new Float32Array(colors.length);
  for (let i = 0; i < segmentIds.length; i++) {
    const seg = segments.find(s => s.id === segmentIds[i]);
    if (seg) {
      const c = new THREE.Color(seg.color);
      segColorArray[i * 3] = c.r;
      segColorArray[i * 3 + 1] = c.g;
      segColorArray[i * 3 + 2] = c.b;
    } else {
      segColorArray[i * 3] = 0.15;
      segColorArray[i * 3 + 1] = 0.15;
      segColorArray[i * 3 + 2] = 0.15;
    }
  }
  
  const segGeometry = geometry.clone();
  segGeometry.setAttribute('color', new THREE.Float32BufferAttribute(segColorArray, 3));
  const segMaterial = new THREE.PointsMaterial({
    size: worldScale * 1.2,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.85,
  });
  const segPoints = new THREE.Points(segGeometry, segMaterial);
  segPoints.name = 'segmentMesh';
  segPoints.visible = true;
  segPoints.position.y = 0.02;
  group.add(segPoints);
  
  return group;
}

// ---- Resampling utilities ----

function resampleFloat32(data, srcW, srcH, dstW, dstH) {
  const result = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(Math.floor(x * xRatio), srcW - 1);
      const sy = Math.min(Math.floor(y * yRatio), srcH - 1);
      result[y * dstW + x] = data[sy * srcW + sx];
    }
  }
  return result;
}

function resampleInt32(data, srcW, srcH, dstW, dstH) {
  const result = new Int32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(Math.floor(x * xRatio), srcW - 1);
      const sy = Math.min(Math.floor(y * yRatio), srcH - 1);
      result[y * dstW + x] = data[sy * srcW + sx];
    }
  }
  return result;
}

function resampleUint8(data, srcW, srcH, dstW, dstH) {
  const result = new Uint8Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(Math.floor(x * xRatio), srcW - 1);
      const sy = Math.min(Math.floor(y * yRatio), srcH - 1);
      result[y * dstW + x] = data[sy * srcW + sx];
    }
  }
  return result;
}

function resampleRGBA(data, srcW, srcH, dstW, dstH) {
  const result = new Uint8ClampedArray(dstW * dstH * 4);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(Math.floor(x * xRatio), srcW - 1);
      const sy = Math.min(Math.floor(y * yRatio), srcH - 1);
      const si = (sy * srcW + sx) * 4;
      const di = (y * dstW + x) * 4;
      result[di] = data[si];
      result[di + 1] = data[si + 1];
      result[di + 2] = data[si + 2];
      result[di + 3] = data[si + 3];
    }
  }
  return result;
}
