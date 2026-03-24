from pathlib import Path
import cv2
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data" / "frames"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helper: readable text
# ------------------------------------------------------------
def put_text(img, text, x, y, color=(255, 255, 255), scale=0.7):
    """Draw text with black outline."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ------------------------------------------------------------
# Preprocess image
# ------------------------------------------------------------
def preprocess(image):
    """
    Light blur before HSV conversion.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return blurred, hsv


# ------------------------------------------------------------
# Green mask
# ------------------------------------------------------------
def make_green_mask(hsv):
    """
    Detect green gate parts.

    These are starting HSV limits. Adjust later if needed.
    """
    lower_green = np.array([35, 40, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove tiny noise and reconnect gate regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    return mask


# ------------------------------------------------------------
# Connected components
# ------------------------------------------------------------
def component_boxes(binary_mask, min_area):
    """
    Return connected component bounding boxes:
    (x, y, w, h, area)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))

    return boxes


# ------------------------------------------------------------
# Choose best component in each quadrant
# ------------------------------------------------------------
def choose_corner_components(boxes, image_shape):
    """
    Pick the strongest green component in each quadrant:
      tl, tr, bl, br

    This matches your real images well because the visible gate
    usually appears as separate green corner pieces.
    """
    h, w = image_shape[:2]

    corners = {
        "tl": None,
        "tr": None,
        "bl": None,
        "br": None,
    }

    scores = {
        "tl": -1,
        "tr": -1,
        "bl": -1,
        "br": -1,
    }

    for box in boxes:
        x, y, bw, bh, area = box
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        if cx < w / 2 and cy < h / 2:
            key = "tl"
        elif cx >= w / 2 and cy < h / 2:
            key = "tr"
        elif cx < w / 2 and cy >= h / 2:
            key = "bl"
        else:
            key = "br"

        # Largest component in each quadrant wins
        if area > scores[key]:
            scores[key] = area
            corners[key] = box

    return corners


# ------------------------------------------------------------
# Estimate gate from corner boxes
# ------------------------------------------------------------
def estimate_gate_from_corners(corners):
    """
    Estimate full gate geometry from green corner components.

    The main assumption:
    - gate frame thickness is roughly constant
    - visible green corner blocks belong to the same square frame

    Returns:
      dict with:
        outer_rect
        opening_rect
        target_center
        thickness
      or None if not enough data
    """
    present = [k for k, v in corners.items() if v is not None]

    # Need at least 3 corners to make a reliable guess
    if len(present) < 3:
        return None

    # Estimate frame thickness from the smaller side of corner boxes
    thickness_values = []
    for box in corners.values():
        if box is None:
            continue
        x, y, w, h, area = box
        thickness_values.append(min(w, h))

    if not thickness_values:
        return None

    thickness = int(np.median(thickness_values))

    # Outer frame limits from visible corner boxes
    left_outer = min(box[0] for key, box in corners.items() if box is not None and key in ("tl", "bl"))
    right_outer = max(box[0] + box[2] for key, box in corners.items() if box is not None and key in ("tr", "br"))
    top_outer = min(box[1] for key, box in corners.items() if box is not None and key in ("tl", "tr"))
    bottom_outer = max(box[1] + box[3] for key, box in corners.items() if box is not None and key in ("bl", "br"))

    # Because the port is roughly square, average width/height into one side estimate
    side_x = right_outer - left_outer
    side_y = bottom_outer - top_outer
    side = int((side_x + side_y) / 2.0)

    # Make the outer frame square-ish
    right_outer = max(right_outer, left_outer + side)
    bottom_outer = max(bottom_outer, top_outer + side)

    # Inner opening is outer frame minus constant thickness on each side
    left_inner = left_outer + thickness
    right_inner = right_outer - thickness
    top_inner = top_outer + thickness
    bottom_inner = bottom_outer - thickness

    if right_inner <= left_inner or bottom_inner <= top_inner:
        return None

    center_x = int((left_inner + right_inner) / 2.0)
    center_y = int((top_inner + bottom_inner) / 2.0)

    return {
        "outer_rect": (
            int(left_outer),
            int(top_outer),
            int(right_outer - left_outer),
            int(bottom_outer - top_outer),
        ),
        "opening_rect": (
            int(left_inner),
            int(top_inner),
            int(right_inner - left_inner),
            int(bottom_inner - top_inner),
        ),
        "target_center": (center_x, center_y),
        "thickness": thickness,
    }


# ------------------------------------------------------------
# Main detection function
# ------------------------------------------------------------
def detect_gate(image):
    """
    Full pipeline:
    1. preprocess
    2. green mask
    3. connected components
    4. choose corner components
    5. estimate gate
    """
    _, hsv = preprocess(image)
    green_mask = make_green_mask(hsv)

    h, w = image.shape[:2]
    min_area = max(500, int(0.003 * h * w))

    boxes = component_boxes(green_mask, min_area=min_area)
    corners = choose_corner_components(boxes, image.shape)
    gate = estimate_gate_from_corners(corners)

    return {
        "green_mask": green_mask,
        "boxes": boxes,
        "corners": corners,
        "gate": gate,
    }


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------
def draw_box(img, box, color, label=None):
    x, y, w, h, area = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        put_text(img, label, x, max(25, y - 8), color, scale=0.6)


def draw_result(image, detection):
    output = image.copy()
    h, w = output.shape[:2]
    image_center = (w // 2, h // 2)

    # Draw image center
    cv2.circle(output, image_center, 7, (255, 0, 0), -1)
    put_text(output, "image center", image_center[0] + 10, image_center[1] - 10, (255, 0, 0))

    corners = detection["corners"]

    # Draw chosen corner components
    if corners["tl"] is not None:
        draw_box(output, corners["tl"], (0, 255, 0), "TL")
    if corners["tr"] is not None:
        draw_box(output, corners["tr"], (0, 255, 255), "TR")
    if corners["bl"] is not None:
        draw_box(output, corners["bl"], (255, 255, 0), "BL")
    if corners["br"] is not None:
        draw_box(output, corners["br"], (255, 0, 255), "BR")

    gate = detection["gate"]

    if gate is None:
        put_text(output, "Gate NOT detected", 20, 35, (0, 0, 255))
        put_text(output, f"green components: {len(detection['boxes'])}", 20, 70, (0, 255, 255))
        put_text(output, f"corners found: {sum(v is not None for v in corners.values())}", 20, 105, (0, 255, 255))
        return output

    # Draw outer frame estimate
    ox, oy, ow, oh = gate["outer_rect"]
    cv2.rectangle(output, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 2)
    put_text(output, "outer frame estimate", ox, max(25, oy - 8), (0, 255, 255), scale=0.6)

    # Draw opening estimate
    ix, iy, iw, ih = gate["opening_rect"]
    cv2.rectangle(output, (ix, iy), (ix + iw, iy + ih), (255, 0, 255), 2)
    put_text(output, "opening estimate", ix, max(25, iy - 8), (255, 0, 255), scale=0.6)

    # Draw target center
    cx, cy = gate["target_center"]
    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
    cv2.line(output, image_center, (cx, cy), (255, 255, 255), 2)

    error_x = cx - image_center[0]
    error_y = cy - image_center[1]

    put_text(output, "Gate detected", 20, 35, (0, 255, 0))
    put_text(output, f"target center: ({cx}, {cy})", 20, 70, (0, 255, 255))
    put_text(output, f"error_x: {error_x}", 20, 105, (0, 255, 255))
    put_text(output, f"error_y: {error_y}", 20, 140, (0, 255, 255))
    put_text(output, f"frame thickness: {gate['thickness']}", 20, 175, (0, 255, 255))
    put_text(output, f"corners found: {sum(v is not None for v in corners.values())}", 20, 210, (0, 255, 255))

    return output


# ------------------------------------------------------------
# Process one image
# ------------------------------------------------------------
def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Could not read: {image_path}")
        return

    detection = detect_gate(image)
    result = draw_result(image, detection)

    cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_green_mask.png"), detection["green_mask"])
    cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_result.png"), result)

    gate = detection["gate"]
    if gate is None:
        print(f"[INFO] {image_path.name}: gate NOT detected")
    else:
        print(f"[INFO] {image_path.name}: target center = {gate['target_center']}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    image_files = sorted(
        list(INPUT_DIR.glob("*.png")) +
        list(INPUT_DIR.glob("*.jpg")) +
        list(INPUT_DIR.glob("*.jpeg"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(image_files)} image(s)")
    print(f"[INFO] Output -> {OUTPUT_DIR}")

    for image_path in image_files:
        process_image(image_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()