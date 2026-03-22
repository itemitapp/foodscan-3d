/**
 * Volume Calculator
 * Calculates volume per ingredient segment using signed tetrahedra method
 * on the displacement mesh geometry
 */

/**
 * Calculate volume for each segment using the signed tetrahedra method
 * The volume of a mesh is computed by summing signed tetrahedra volumes
 * formed between each triangle face and the origin.
 * 
 * V = (1/6) * Σ |v1 · (v2 × v3)| for each triangle (v1, v2, v3)
 * 
 * @param {THREE.Group} group - the mesh group from mesh-generator
 * @param {number} plateDiameter - plate diameter in cm
 * @returns {object} map of segmentId -> volume in ml (cm³)
 */
export function calculateVolumes(group) {
  const foodMesh = group.getObjectByName('foodMesh');
  if (!foodMesh) return {};
  
  const geometry = foodMesh.geometry;
  const positions = geometry.getAttribute('position').array;
  const segmentIds = geometry.getAttribute('segmentId').array;
  const indexAttr = geometry.getIndex();
  const segments = foodMesh.userData.segments;
  
  // Point cloud fallback: estimate volume from bounding box
  if (!indexAttr) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const totalVol = (box.max.x - box.min.x) * (box.max.y - box.min.y) * (box.max.z - box.min.z);
    
    // Count points per segment
    const segCounts = {};
    let totalPoints = 0;
    segments.forEach(s => { segCounts[s.id] = 0; });
    for (let i = 0; i < segmentIds.length; i++) {
      const sid = segmentIds[i];
      if (segCounts[sid] !== undefined) segCounts[sid]++;
      totalPoints++;
    }
    
    const segmentVolumes = {};
    segments.forEach(s => {
      segmentVolumes[s.id] = totalPoints > 0 ? (totalVol * segCounts[s.id] / totalPoints) : 0;
    });
    return segmentVolumes;
  }
  
  const index = indexAttr.array;
  
  // Calculate volume per segment
  // For each triangle, compute the signed volume of the tetrahedron
  // formed by the triangle and the base plane (y=0)
  const segmentVolumes = {};
  segments.forEach(s => { segmentVolumes[s.id] = 0; });
  
  for (let i = 0; i < index.length; i += 3) {
    const a = index[i];
    const b = index[i + 1];
    const c = index[i + 2];
    
    // Get the primary segment for this face (majority vote of vertices)
    const segA = segmentIds[a];
    const segB = segmentIds[b];
    const segC = segmentIds[c];
    
    // Majority vote
    let faceSeg;
    if (segA === segB || segA === segC) faceSeg = segA;
    else if (segB === segC) faceSeg = segB;
    else faceSeg = segA;
    
    if (faceSeg < 0) continue; // skip background
    
    // Get vertex positions
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
    const cx = positions[c * 3], cy = positions[c * 3 + 1], cz = positions[c * 3 + 2];
    
    // For a displacement surface, volume under the surface is calculated
    // by integrating the height. We use a prism-based approach:
    // For each triangle, compute the average height and multiply by the
    // projected area on the XZ plane
    
    const avgHeight = (ay + by + cy) / 3;
    
    // Projected area on XZ plane using cross product
    // Area = 0.5 * |(B-A) × (C-A)| projected onto XZ
    const abx = bx - ax, abz = bz - az;
    const acx = cx - ax, acz = cz - az;
    const projectedArea = Math.abs(abx * acz - abz * acx) / 2;
    
    // Volume contribution = projected area * average height
    const vol = projectedArea * avgHeight;
    
    if (segmentVolumes[faceSeg] !== undefined) {
      segmentVolumes[faceSeg] += vol;
    }
  }
  
  // Convert from cm³ to ml (1:1) and ensure positive
  const result = {};
  segments.forEach(s => {
    result[s.id] = Math.abs(segmentVolumes[s.id] || 0);
  });
  
  return result;
}

/**
 * Format volume for display with multiple units
 * @param {number} volumeMl - volume in milliliters
 * @returns {object} formatted values
 */
export function formatVolume(volumeMl) {
  return {
    ml: volumeMl.toFixed(1),
    cups: (volumeMl / 236.588).toFixed(2),
    tbsp: (volumeMl / 14.787).toFixed(1),
    floz: (volumeMl / 29.5735).toFixed(1),
    liters: (volumeMl / 1000).toFixed(3),
  };
}

/**
 * Estimate approximate calories based on volume and food type heuristic
 * Very rough estimates for display purposes
 */
const CALORIE_DENSITY = {
  'White/Rice': 1.3,      // kcal per ml
  'Brown/Meat': 2.5,
  'Green/Vegetable': 0.25,
  'Red/Tomato': 0.18,
  'Orange/Carrot': 0.35,
  'Yellow/Corn': 0.9,
  'Dark/Sauce': 0.8,
  'Beige/Bread': 2.5,
  'Pink/Salmon': 2.0,
  'Blue/Berry': 0.4,
  'Purple/Eggplant': 0.25,
  'Cyan/Herb': 0.2,
  'Other': 1.0,
};

export function estimateCalories(volumeMl, label) {
  // Strip number suffix for lookup
  const baseLabel = label.replace(/\s+\d+$/, '');
  const density = CALORIE_DENSITY[baseLabel] || 1.0;
  return Math.round(volumeMl * density);
}
