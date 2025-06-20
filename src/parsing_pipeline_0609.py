import numpy as np, matplotlib.pyplot as re, sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from modules.hisam.inference  import HiSam_Inference
from modules.ocr.main         import PaddleOCRClient
from modules.textremover.lama import LaMa
from modules.layout.yolo      import Yolo_Client
from ultralytics              import SAM

if len(sys.argv) < 2:
    print("사용법: python parsing_pipeline_0609.py /abs/path/to/folder")
    sys.exit(1)

FOLDER_PATH = Path(sys.argv[1]).expanduser().resolve()
if not FOLDER_PATH.is_dir():
    print(f"[ERROR] {FOLDER_PATH} 는 폴더가 아닙니다.")
    sys.exit(1)

DILATE_PX = 2
WEIGHTS   = Path('../weights')            # 가중치 경로 (필요 시 절대경로로 수정)
OUT_ROOT  = Path('./parsing_outputs')
OUT_ROOT.mkdir(exist_ok=True)

# PaddleOCR: Text 추출
def get_text(it):
    for k in ('text','label','sentence','words'):
        if k in it and it[k]: return str(it[k])
    return ''

# PaddleOCR: Text 전처리
def sanitize(txt:str, fallback:str):
    fname = re.sub(r"[\\/:*?\"<>|\s]+", "_", txt).strip("._")
    return fname or fallback

def process_image(img_path: Path):
    """하나의 이미지에 대해 파이프라인 수행 & 결과 저장"""
    img = Image.open(img_path).convert('RGB')

    # 0) 출력 하위 폴더 준비
    sub_dir  = OUT_ROOT / img_path.stem
    text_dir = sub_dir / "text_layers"
    obj_dir  = sub_dir / "object_layers"
    bg_dir   = sub_dir / "background"
    
    sub_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # 1) Hi-SAM 텍스트 마스크
    _, hisam_mask = hisam(input_img=img)
    hisam_mask = hisam_mask.point(lambda p:255 if p>=128 else 0).convert('L')
    hisam_bin  = np.array(hisam_mask, dtype=np.uint8) > 0

    # 2) OCR
    items, _ = ocr.run_ocr(img)
    img_np   = np.array(img)
    if img_np.ndim == 2:                    # grayscale 보정
        img_np = np.stack([img_np]*3, -1)
    h_np, w_np = img_np.shape[:2]
    ocr_union  = np.zeros((h_np, w_np), dtype=np.uint8)

    saved_txt_layers = 0
    for idx, it in enumerate(items):
        mask = it.get("polygon")
        if mask is None: continue
        mask_np = np.array(mask, dtype=np.uint8) > 0
        ocr_union |= mask_np
        if mask_np.sum() == 0: continue

        ys, xs = np.where(mask_np)
        y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        roi_rgb   = img_np[y0:y1, x0:x1]
        roi_alpha = mask_np[y0:y1, x0:x1].astype(np.uint8)*255
        rgba      = np.dstack([roi_rgb, roi_alpha])
        crop_rgba = Image.fromarray(rgba, "RGBA")

        txt        = get_text(it) or f"ocr_{idx:03d}"
        fname_base = sanitize(txt, f"ocr_{idx:03d}")
        save_path  = text_dir / f"{fname_base}.png"
        if save_path.exists():
            save_path = text_dir / f"{fname_base}_{idx}.png"
        crop_rgba.save(save_path)
        saved_txt_layers += 1

    # 3) 텍스트 제거 마스크 & 라마
    text_mask = (ocr_union & hisam_bin).astype(np.uint8)*255
    text_mask_img = Image.fromarray(text_mask, "L")
    text_removed, _, _ = lama.remove_text_by_mask(img, text_mask_img)
    text_removed = text_removed.convert('RGB')

    # 4) YOLO + SAM
    bboxes, classes, yolo_vis = yolo.execute(text_removed)

    obj_bb = [b for b, c in zip(bboxes, classes) if c == 0]
    pic_bb = [b for b, c in zip(bboxes, classes) if c == 1]


    # Fore-Ground Mask { Object, Picture }
    fg = np.zeros((h_np, w_np), dtype=np.uint8)
    if obj_bb:
        res = sam_seg(text_removed, bboxes=obj_bb)
        for m in res[0].masks:
            fg |= m.data.squeeze().cpu().numpy()
    for b in pic_bb:
        x1,y1,x2,y2=b
        fg[y1:y2,x1:x2]=1

    for idx, bbox in enumerate(obj_bb):
        x1,y1,x2,y2 = map(int, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w_np,x2), min(h_np,y2)
        if x2<=x1 or y2<=y1: continue
        crop = text_removed.crop((x1,y1,x2,y2))
        crop.save(obj_dir / f"object_{idx:03d}.png")

    for idx, bbox in enumerate(pic_bb):
        x1,y1,x2,y2 = map(int, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w_np,x2), min(h_np,y2)
        if x2<=x1 or y2<=y1: continue
        crop = text_removed.crop((x1,y1,x2,y2))
        crop.save(obj_dir / f"picture_{idx:03d}.png")

    # 5) 배경 추출 (선택)
    fg_pil = Image.fromarray(fg*255, mode='L')
    bg, _, _ = lama.remove_text_by_mask(text_removed, fg_pil)
    bg = bg.convert('RGB')
    bg_path = bg_dir / "background.png"
    bg.save(bg_path)

    # 6) 요약 PNG 시각화 (옵션)
    try:
        import matplotlib.pyplot as plt
        panels=[('Original',img),('Text Mask (Hi-SAM & Paddle-OCR)',text_mask_img),
            ('Text Removed',text_removed),('YOLO',yolo_vis),
            ('ForeGround Mask',fg_pil),('Background',bg)]
        fig,ax=plt.subplots(2,3,figsize=(12,8))
        for (t,im),a in zip(panels,ax.ravel()):
            a.imshow(im, cmap='gray' if im.mode=='L' else None)
            a.set_title(t); a.axis('off')
        plt.tight_layout(); 
        plt.savefig(sub_dir/'summary.png'); 
        plt.close(fig)

    except Exception as e:
        print(f"[WARN] summary 시각화 실패: {e}")

    print(f"📂 {img_path.name} → {text_dir.relative_to(OUT_ROOT)} 저장 완료.")


print("📦  Loading models (Hi-SAM, Paddle-OCR, LaMA, YOLO, SAM)")
hisam   = HiSam_Inference(check_point_dir=WEIGHTS, model_path='sam_tss_h_textseg.pth')
ocr     = PaddleOCRClient()
lama    = LaMa(model_path=f"{WEIGHTS}/big-lama.pt")
yolo    = Yolo_Client(model_path=f'{WEIGHTS}/yolov11.pt')
sam_seg = SAM(f'{WEIGHTS}/sam2_b.pt')
print("✅  All models loaded.\n")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
img_files  = sorted([p for p in FOLDER_PATH.rglob('*') if p.suffix.lower() in IMAGE_EXTS])

print(f"\n🚀  {FOLDER_PATH} 내 이미지 {len(img_files)}장 처리 시작 …\n")
for img_path in tqdm(img_files, desc="Processing Images"):
    try:
        process_image(img_path)
    except Exception as e:
        print(f"[ERROR] {img_path.name} 처리 실패: {e}")

print("\n🎉  전체 작업 완료!")