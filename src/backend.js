/**
 * FoodScan 3D — Local Backend API Module
 * Communicates with the Python MapAnything backend running at localhost:8000.
 */

const BACKEND_URL = 'http://localhost:8000';
let backendAvailable = null; // null = not checked, true/false = checked

/**
 * Check if the local backend is running. Caches the result.
 * Returns { available: bool, device: string }
 */
export async function checkBackend() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`, { signal: AbortSignal.timeout(1500) });
    if (!res.ok) throw new Error('not ok');
    const data = await res.json();
    backendAvailable = true;
    return { available: true, device: data.device, version: data.version };
  } catch {
    backendAvailable = false;
    return { available: false };
  }
}

export function isBackendAvailable() {
  return backendAvailable === true;
}

/**
 * Send images to the backend for metric 3D reconstruction.
 * @param {File[]} imageFiles - Array of File objects
 * @param {number} plateDiameterCm - Plate diameter in centimeters
 * @param {function(string)} onStatus - Status update callback
 * @param {object} config - MapAnything configuration overrides
 * @returns {Promise<{glb_url, total_volume_ml, segments, job_id}>}
 */
export async function reconstructWithBackend(imageFiles, plateDiameterCm, onStatus, config = {}) {
  const form = new FormData();
  for (const f of imageFiles) {
    form.append('images', f);
  }
  form.append('plate_diameter_cm', String(plateDiameterCm));

  // MapAnything config params
  form.append('segments',           String(config.segments           ?? 0));
  form.append('min_food_height_mm', String(config.minFoodHeightMm   ?? 3.0));
  form.append('memory_efficient',   String(config.memoryEfficient    ?? true));
  form.append('minibatch_size',     String(config.minibatchSize      ?? 1));
  form.append('mask_edges',         String(config.maskEdges          ?? true));

  onStatus?.(`Sending ${imageFiles.length} photo(s) to local AI engine...`);

  const res = await fetch(`${BACKEND_URL}/api/reconstruct`, {
    method: 'POST',
    body: form,
    // No timeout: can take minutes for model download + inference
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Backend error');
  }

  const result = await res.json();
  onStatus?.(`Reconstruction complete on ${result.device_used.toUpperCase()}`);
  return result;
}

/**
 * Build the full URL to load a GLB file from the backend.
 */
export function getGlbUrl(glbPath) {
  return `${BACKEND_URL}${glbPath}`;
}
