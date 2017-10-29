# Import relevant modules
import sys
import numpy as np
import cv2

# Define points
pts = np.array([[542, 107], [562, 102], [582, 110], [598, 142], [600, 192], [601, 225], [592, 261], [572, 263], [551, 245], [526, 220], [520, 188], [518, 152], [525, 127], [524, 107]], dtype=np.int32)

### Define image here
img = 255*np.ones((300, 700, 3), dtype=np.uint8)

refPt = []
done = False
mousex = 0
mousey = 0

def point_and_shoot(event, x, y, flags, param):
    global refPt, done, mousex, mousey
    mousex = x
    mousey = y
    """
    right-click = point
    left-click = draw shape

    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if done:
            refPt = []
            done = False
        refPt.append((x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        done = True

cv2.namedWindow("Output Image", flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Output Image", point_and_shoot)

alpha = 0.5

# Show image, wait for user input, then save the image
while True:
    # Initialize mask
    mask = np.zeros((img.shape[0], img.shape[1]))

    # Create output image (untranslated)
    out_img = img.copy()
    out = np.zeros_like(img)

    if len(refPt) > 0:
        cv2.polylines(out_img, np.array([refPt]), False, (255,0,0), 1)
        if not done:
            cv2.line(out_img, refPt[-1], (mousex,mousey), (0,255,0))

    if done and refPt > 2:
        # Create mask that defines the polygon of points
        points = np.array(refPt, dtype=np.int32) 
        cv2.fillConvexPoly(mask, points, 1)
        mask = mask.astype(np.bool)


        out[mask] = img[mask]

        cv2.addWeighted(out, alpha, out_img, 1-alpha, 0, out_img)

        # Determine top left and bottom right of translated image
        topleft = points.min(axis=0)
        bottomright = points.max(axis=0)

        # Draw rectangle
        cv2.rectangle(out_img, tuple(topleft), tuple(bottomright), color=(255,0,0))

        #out = out_translate


    cv2.imshow('Output Image', out_img)


    key =cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
        cv2.destroyAllWindows()
	sys.exit(0)
