import cv2
import numpy as np

# =========================
video_path = r"C:/Users/Pakhi/OneDrive/Documents/PGIMER_2025/Ongoing_Projects/PROSPECT/Prelim_Model/Data/Normal/MOV_$$00000019C2CB25_20251211114049.avi"
out_path   = r"C:/Users/Pakhi/OneDrive/Documents/PGIMER_2025/Ongoing_Projects/PROSPECT/Prelim_Model/Data/Normal/Adaptive_crop_minimal.avi"
# =========================

LEFT_UI, TOP_UI, BOTTOM_UI = 120, 80, 90
MARGIN = 15
ALPHA = 0.90
MAX_SCALE = 0.04

def preprocess(frame):
    h, w = frame.shape[:2]

    # remove fixed UI
    f = frame[TOP_UI:h-BOTTOM_UI, LEFT_UI:w]

    # trim black sides
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cols = np.where(g.sum(0) > 5)[0]
    if len(cols):
        f = f[:, cols[0]:cols[-1]]

    return f


def largest_mask(gray):
    T = np.percentile(gray, 15)
    m = (gray > T).astype(np.uint8)*255

    k = np.ones((9,9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)

    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return None

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


def smooth(prev, curr):
    if prev is None:
        return curr

    sm = [int(ALPHA*p + (1-ALPHA)*c) for p,c in zip(prev,curr)]

    pw, ph = prev[2]-prev[0], prev[3]-prev[1]
    cw, ch = sm[2]-sm[0], sm[3]-sm[1]

    if abs(cw-pw) > pw*MAX_SCALE or abs(ch-ph) > ph*MAX_SCALE:
        return prev
    return sm


def pad(img, H, W):
    h,w = img.shape[:2]
    t,b = (H-h)//2, H-h-(H-h)//2
    l,r = (W-w)//2, W-w-(W-w)//2
    return cv2.copyMakeBorder(img,t,b,l,r,cv2.BORDER_CONSTANT,0)


# ---------- video ----------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

prev_bbox = None
writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = largest_mask(gray)
    if mask is None:
        continue

    ys, xs = np.where(mask)
    y1,y2 = ys.min(), ys.max()
    x1,x2 = xs.min(), xs.max()

    y1,x1 = max(0,y1-MARGIN), max(0,x1-MARGIN)
    y2,x2 = min(frame.shape[0],y2+MARGIN), min(frame.shape[1],x2+MARGIN)

    bbox = smooth(prev_bbox, [x1,y1,x2,y2])
    prev_bbox = bbox

    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if writer is None:
        H,W = crop.shape[:2]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W,H))

    crop = pad(crop, H, W)
    writer.write(crop)

cap.release()
writer.release()
