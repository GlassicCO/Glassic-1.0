import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from simple_lama_inpainting import SimpleLama

print("🚀 Booting Glassic v4 Engine Core (FastAPI Edition)...")
app = FastAPI(title="Glassic API")

# Allow the mobile app to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lama = SimpleLama()
print("✅ LaMa Generative AI Loaded Successfully!")

def run_safe_lama(img_bgr, mask_cv):
    h, w = img_bgr.shape[:2]
    max_dim = 1024 
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        proc_w, proc_h = int(w * scale), int(h * scale)
    else:
        proc_w, proc_h = w, h

    proc_bgr = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    proc_mask = cv2.resize(mask_cv, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

    img_pil = Image.fromarray(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(proc_mask).convert('L')
    
    result_pil = lama(img_pil, mask_pil)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return result_bgr

@app.post("/retouch")
async def process_image(
    image: UploadFile = File(...), 
    mask: UploadFile = File(...), 
    feature: str = Form(...)
):
    print(f"⚙️ Executing module: {feature}")

    try:
        # 1. Read the incoming image from the mobile app
        img_bytes = await image.read()
        bg_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bg_pil = ImageOps.exif_transpose(bg_pil)
        img_bgr = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

        # 2. Read the incoming mask
        mask_bytes = await mask.read()
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        # Ensure mask is exactly the same size as the image
        mask_pil = mask_pil.resize((img_bgr.shape[1], img_bgr.shape[0]), Image.NEAREST)
        mask_cv = np.array(mask_pil)
        
        # Threshold the mask to pure black/white
        _, mask_cv = cv2.threshold(mask_cv, 10, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(mask_cv) == 0:
            raise HTTPException(status_code=400, detail="Mask is empty. Paint over the target area.")

        # Establish smooth blending properties
        kernel_size = max(21, int(max(img_bgr.shape[:2]) * 0.02) | 1)
        feather_mask = cv2.GaussianBlur(mask_cv, (kernel_size, kernel_size), 0)
        blend_weights = cv2.cvtColor(feather_mask, cv2.COLOR_GRAY2BGR) / 255.0

        output_bgr = img_bgr.copy()

        # ==========================================
        # FEATURE 1: SMART REVERSE EVERYTHING
        # ==========================================
        if feature == "Reverse Text & Area":
            x, y, w, h = cv2.boundingRect(mask_cv)
            pad = 15
            x_start, y_start = max(0, x - pad), max(0, y - pad)
            x_end, y_end = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
            
            roi = img_bgr[y_start:y_end, x_start:x_end].copy()
            flipped_roi = cv2.flip(roi, 1)
            
            canvas = img_bgr.copy()
            canvas[y_start:y_end, x_start:x_end] = flipped_roi
            output_bgr = (canvas * blend_weights + img_bgr * (1.0 - blend_weights)).astype(np.uint8)

        # ==========================================
        # FEATURE 2: REMOVE GRIME & SMUDGES
        # ==========================================
        elif feature == "Remove Grime & Smudges":
            grime_mask = cv2.dilate(mask_cv, np.ones((7, 7), np.uint8), iterations=2)
            inpainted = run_safe_lama(img_bgr, grime_mask)
            output_bgr = (inpainted * blend_weights + img_bgr * (1.0 - blend_weights)).astype(np.uint8)

        # ==========================================
        # FEATURE 3: REMOVE GLARE & FLASH
        # ==========================================
        elif feature == "Remove Glare":
            glare_mask = cv2.dilate(mask_cv, np.ones((15, 15), np.uint8), iterations=3)
            inpainted = run_safe_lama(img_bgr, glare_mask)
            output_bgr = (inpainted * blend_weights + img_bgr * (1.0 - blend_weights)).astype(np.uint8)

        # ==========================================
        # FEATURE 4: NATURAL LIGHT-AWARE SKIN GLOW
        # ==========================================
        elif feature == "Skin Glow":
            smoothed = cv2.bilateralFilter(img_bgr, d=11, sigmaColor=70, sigmaSpace=70)
            hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            v_8u = v.astype(np.uint8)
            masked_v = cv2.bitwise_and(v_8u, mask_cv)
            mean_val = cv2.mean(masked_v, mask=mask_cv)[0]
            
            _, hotspots = cv2.threshold(masked_v, min(255, int(mean_val * 1.1)), 255, cv2.THRESH_BINARY)
            blur_size = max(41, int(max(img_bgr.shape[:2]) * 0.05) | 1)
            soft_hotspots = cv2.GaussianBlur(hotspots, (blur_size, blur_size), 0).astype(np.float32) / 255.0
            
            v_boost = v * (1.05 + (0.07 * soft_hotspots)) 
            s_boost = s * 1.03
            
            v_final = np.clip(v_boost, 0, 255).astype(np.uint8)
            s_final = np.clip(s_boost, 0, 255).astype(np.uint8)
            h_final = h.astype(np.uint8)
            
            glow_hsv = cv2.merge((h_final, s_final, v_final))
            shiny_skin = cv2.cvtColor(glow_hsv, cv2.COLOR_HSV2BGR)
            
            output_bgr = (shiny_skin * blend_weights + img_bgr * (1.0 - blend_weights)).astype(np.uint8)

        # Encode back to an image file to send to the phone
        final_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        final_pil = Image.fromarray(final_rgb)
        
        img_byte_arr = io.BytesIO()
        final_pil.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        print("✅ Processing Complete! Sending back to mobile.")
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
