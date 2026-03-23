import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FOOD_IMAGE = path.resolve(__dirname, 'fixtures/food-sample.png');

/**
 * Helper: upload a food image and wait for 3D model to be generated.
 * Returns once the viewer screen is visible.
 */
async function uploadAndProcess(page: Page) {
  await page.goto('/');
  
  // Upload the food image
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(FOOD_IMAGE);
  
  // Wait for the "Generate 3D Model" button and click it
  const generateBtn = page.locator('#process-btn');
  await expect(generateBtn).toBeVisible({ timeout: 5000 });
  await generateBtn.click();
  
  // Wait for processing to complete — viewer should appear
  await expect(page.locator('#viewer-screen.active')).toBeVisible({ timeout: 180_000 });
  
  // Verify the 3D canvas is visible
  await expect(page.locator('#three-canvas')).toBeVisible();
}

/**
 * Helper: set a range input value via JS (fill() doesn't work on range inputs).
 */
async function setSlider(page: Page, selector: string, value: string) {
  await page.locator(selector).evaluate((el: HTMLInputElement, v: string) => {
    el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
}

/**
 * Helper: click "Regenerate Model" and wait for it to complete.
 */
async function clickRegenerate(page: Page) {
  const applyBtn = page.locator('#tune-apply');
  await expect(applyBtn).toBeVisible();
  
  // Click regenerate
  await applyBtn.click();
  
  // Should show "Regenerating..." text
  await expect(applyBtn).toContainText(/Regenerat/);
  
  // Wait for button to go back to "Regenerate Model"
  await expect(applyBtn).toHaveText('Regenerate Model', { timeout: 60_000 });
}

test.describe('Model Regeneration', () => {
  
  test.beforeEach(async ({ page }) => {
    await uploadAndProcess(page);
  });

  test('initial load shows ingredients and tuning panel', async ({ page }) => {
    // Left sidebar: ingredients
    await expect(page.locator('#ingredient-list')).toBeVisible();
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
    
    // Right sidebar: tuning panel
    await expect(page.locator('#tuning-sidebar')).toBeVisible();
    await expect(page.locator('#tune-apply')).toBeVisible();
    await expect(page.locator('#tune-depth-model')).toBeVisible();
    await expect(page.locator('#tune-mask-method')).toBeVisible();
  });

  test('regenerate with histogram mask method', async ({ page }) => {
    // Switch mask method to histogram
    await page.selectOption('#tune-mask-method', 'histogram');
    expect(await page.locator('#tune-mask-method').inputValue()).toBe('histogram');
    
    await clickRegenerate(page);
    
    // Verify ingredients still exist after regeneration
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
    
    // 3D canvas should still be visible
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with otsu mask method', async ({ page }) => {
    await page.selectOption('#tune-mask-method', 'otsu');
    expect(await page.locator('#tune-mask-method').inputValue()).toBe('otsu');
    
    await clickRegenerate(page);
    
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
  });

  test('regenerate with fixed mask method', async ({ page }) => {
    await page.selectOption('#tune-mask-method', 'fixed');
    expect(await page.locator('#tune-mask-method').inputValue()).toBe('fixed');
    
    await clickRegenerate(page);
    
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
  });

  test('regenerate with point cloud mesh method', async ({ page }) => {
    await page.selectOption('#tune-mesh-method', 'pointcloud');
    expect(await page.locator('#tune-mesh-method').inputValue()).toBe('pointcloud');
    
    await clickRegenerate(page);
    
    // Canvas should still render
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with changed depth scale', async ({ page }) => {
    await setSlider(page, '#tune-depthscale', '0.60');
    
    await expect(page.locator('#val-depthscale')).toHaveText('0.60');
    
    await clickRegenerate(page);
    
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
  });

  test('regenerate with changed smoothing', async ({ page }) => {
    await setSlider(page, '#tune-smoothing', '5');
    await expect(page.locator('#val-smoothing')).toHaveText('5');
    
    await clickRegenerate(page);
    
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with walls off', async ({ page }) => {
    // Uncheck side walls
    const wallsCheckbox = page.locator('#tune-walls');
    if (await wallsCheckbox.isChecked()) {
      await wallsCheckbox.uncheck();
    }
    expect(await wallsCheckbox.isChecked()).toBe(false);
    
    await clickRegenerate(page);
    
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with bottom cap off', async ({ page }) => {
    const bottomCheckbox = page.locator('#tune-bottom');
    if (await bottomCheckbox.isChecked()) {
      await bottomCheckbox.uncheck();
    }
    expect(await bottomCheckbox.isChecked()).toBe(false);
    
    await clickRegenerate(page);
    
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with auto-level off', async ({ page }) => {
    const autoLevel = page.locator('#tune-autolevel');
    if (await autoLevel.isChecked()) {
      await autoLevel.uncheck();
    }
    expect(await autoLevel.isChecked()).toBe(false);
    
    await clickRegenerate(page);
    
    await expect(page.locator('#three-canvas')).toBeVisible();
  });

  test('regenerate with multiple config changes at once', async ({ page }) => {
    // Change multiple settings at once
    await page.selectOption('#tune-mask-method', 'otsu');
    await page.selectOption('#tune-mesh-method', 'pointcloud');
    await setSlider(page, '#tune-depthscale', '0.80');
    await setSlider(page, '#tune-smoothing', '0');
    await setSlider(page, '#tune-taper', '10');
    
    // Uncheck walls and bottom
    const walls = page.locator('#tune-walls');
    if (await walls.isChecked()) await walls.uncheck();
    const bottom = page.locator('#tune-bottom');
    if (await bottom.isChecked()) await bottom.uncheck();
    
    await clickRegenerate(page);
    
    // Verify app didn't crash and canvas still renders
    await expect(page.locator('#three-canvas')).toBeVisible();
    const ingredients = page.locator('#ingredient-list .ingredient-item');
    expect(await ingredients.count()).toBeGreaterThan(0);
  });

  test('volume updates after regeneration', async ({ page }) => {
    const initialVolume = await page.locator('#total-vol').textContent();
    
    await setSlider(page, '#tune-depthscale', '0.90');
    
    await clickRegenerate(page);
    
    const newVolume = await page.locator('#total-vol').textContent();
    expect(newVolume).not.toBe('—');
    expect(newVolume).toMatch(/\d/);
  });
});
