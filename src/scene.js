/**
 * Three.js Scene Manager
 * Sets up camera, lights, controls, and raycasting
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class SceneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.foodGroup = null;
    this.selectedSegmentId = null;
    this.onSegmentClick = null;
    
    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x12121a, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    // Camera
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    this.camera.position.set(0, 20, 25);
    this.camera.lookAt(0, 0, 0);
    
    // Controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 80;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.3;
    this.controls.target.set(0, 2, 0);
    
    // Raycaster
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    
    // Lights
    this.setupLights();
    
    // Grid helper (subtle)
    const gridHelper = new THREE.GridHelper(50, 50, 0x1a1a2e, 0x1a1a2e);
    gridHelper.position.y = -0.2;
    this.scene.add(gridHelper);
    
    // Handle resize
    this.resize();
    window.addEventListener('resize', () => this.resize());
    
    // Click handling
    canvas.addEventListener('click', (e) => this.handleClick(e));
    canvas.addEventListener('mousemove', (e) => this.handleHover(e));
    
    // Start render loop
    this.animate();
  }
  
  setupLights() {
    // Ambient
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambient);
    
    // Key light
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
    keyLight.position.set(10, 20, 10);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    this.scene.add(keyLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0x6366f1, 0.3);
    fillLight.position.set(-10, 10, -5);
    this.scene.add(fillLight);
    
    // Rim light
    const rimLight = new THREE.DirectionalLight(0xa855f7, 0.2);
    rimLight.position.set(0, 5, -15);
    this.scene.add(rimLight);
    
    // Hemisphere light for natural feel
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1a1a2e, 0.3);
    this.scene.add(hemiLight);
  }
  
  resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
  
  setFoodModel(group) {
    // Remove old model
    if (this.foodGroup) {
      this.scene.remove(this.foodGroup);
    }
    
    this.foodGroup = group;
    this.scene.add(group);
    
    // Auto-fit camera
    const box = new THREE.Box3().setFromObject(group);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    const cameraZ = maxDim / (2 * Math.tan(fov / 2)) * 1.5;
    
    this.camera.position.set(center.x, center.y + maxDim * 0.8, center.z + cameraZ);
    this.controls.target.copy(center);
    this.controls.update();
  }
  
  handleClick(event) {
    if (!this.foodGroup || !this.onSegmentClick) return;
    
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    this.raycaster.setFromCamera(this.mouse, this.camera);
    
    // Raycast against the food mesh and segment mesh
    const meshes = [];
    this.foodGroup.traverse(child => {
      if (child.isMesh && (child.name === 'foodMesh' || child.name === 'segmentMesh')) {
        meshes.push(child);
      }
    });
    
    const intersects = this.raycaster.intersectObjects(meshes);
    
    if (intersects.length > 0) {
      const intersection = intersects[0];
      const face = intersection.face;
      if (face) {
        const foodMesh = this.foodGroup.getObjectByName('foodMesh');
        const segIds = foodMesh.geometry.getAttribute('segmentId').array;
        const segId = segIds[face.a];
        
        if (segId >= 0) {
          this.selectedSegmentId = segId;
          this.onSegmentClick(segId);
        }
      }
    } else {
      // Clicked empty space - deselect
      this.selectedSegmentId = null;
      this.onSegmentClick(null);
    }
  }
  
  handleHover(event) {
    if (!this.foodGroup) return;
    this.canvas.style.cursor = 'default';
    
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    this.raycaster.setFromCamera(this.mouse, this.camera);
    
    const meshes = [];
    this.foodGroup.traverse(child => {
      if (child.isMesh && (child.name === 'foodMesh' || child.name === 'segmentMesh')) {
        meshes.push(child);
      }
    });
    
    const intersects = this.raycaster.intersectObjects(meshes);
    if (intersects.length > 0) {
      this.canvas.style.cursor = 'pointer';
    }
  }
  
  toggleWireframe(show) {
    if (!this.foodGroup) return;
    const wireframe = this.foodGroup.getObjectByName('wireframe');
    if (wireframe) wireframe.visible = show;
  }
  
  toggleSegments(show) {
    if (!this.foodGroup) return;
    const segMesh = this.foodGroup.getObjectByName('segmentMesh');
    const foodMesh = this.foodGroup.getObjectByName('foodMesh');
    if (segMesh) segMesh.visible = show;
    if (foodMesh) foodMesh.visible = !show;
  }
  
  resetView() {
    if (!this.foodGroup) return;
    const box = new THREE.Box3().setFromObject(this.foodGroup);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    const cameraZ = maxDim / (2 * Math.tan(fov / 2)) * 1.5;
    
    this.camera.position.set(center.x, center.y + maxDim * 0.8, center.z + cameraZ);
    this.controls.target.copy(center);
  }
  
  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
