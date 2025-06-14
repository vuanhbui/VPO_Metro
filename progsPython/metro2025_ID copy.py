# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:17:49 2025
@author: fross
"""

import os
import numpy as np
import joblib
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


# Globals for training
hog_features = []
hog_labels   = []
svm_model_path = 'svm_model.pkl'
svm = None

# Segmentation parameters
area_min = 3000            # minimal area
circ_thresh = 0.85         # minimal circularity
neg_samples_per_gt = 2     # negative samples per GT in Learn

# IoU & F1 helpers

def draw_rectangle(x1,x2,y1,y2,color):
    rect = Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=2, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    interA = interW*interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interA/float(areaA+areaB-interA) if (areaA+areaB-interA)>0 else 0

def evaluate_f1(preds, gts, iou_thresh=0.5):
    TP=FP=0
    matched=set()
    for i,pb in enumerate(preds):
        best_i=0; best_j=None
        for j,gb in enumerate(gts):
            val = iou(pb,gb)
            if val>best_i:
                best_i, best_j = val, j
        if best_i>=iou_thresh and best_j not in matched:
            TP+=1; matched.add(best_j)
        else:
            FP+=1
    FN = len(gts)-len(matched)
    prec = TP/(TP+FP) if TP+FP else 0
    rec  = TP/(TP+FN) if TP+FN else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return prec, rec, f1



def evaluate_full_test():
    # Load GT for Test
    GTt = sio.loadmat('progsPython/Test.mat')['BD']
    test_ids = np.arange(1,262)[np.arange(1,262)%3!=0]

    # 1) Segmentation-only evaluation
    all_preds = []
    all_gts   = []
    for n in test_ids:
        im = cv2.imread(f"BD_METRO/IM ({n}).JPG")[...,::-1] / 255.0
        preds = segment_regions(im)
        gts   = [(int(g[3]), int(g[1]), int(g[4]), int(g[2])) for g in GTt if g[0]==n]
        all_preds.extend(preds)
        all_gts.extend(gts)
    p_s, r_s, f_s = evaluate_f1(all_preds, all_gts)
    print(f"Segmentation Test → P:{p_s:.3f}, R:{r_s:.3f}, F1:{f_s:.3f}")

    # 2) Classification-only evaluation
    TP = FP = FN = 0
    for n in test_ids:
        im = cv2.imread(f"BD_METRO/IM ({n}).JPG")[...,::-1] / 255.0
        preds = segment_regions(im)
        gts = [g for g in GTt if g[0]==n]
        matched = set()
        for pb in preds:
            best_i = 0; best_gt = None
            for g in gts:
                gb = (int(g[3]), int(g[1]), int(g[4]), int(g[2]))
                val = iou(pb, gb)
                if val > best_i:
                    best_i, best_gt = val, g
            if best_i is not None and best_i >= 0.5 and best_gt is not None:
                # create a hashable key for the GT
                best_key = tuple(int(x) for x in best_gt)
                if best_key not in matched:
                    # classify correctly localized ROI
                    y1, y2, x1, x2, cls_true = int(best_gt[1]), int(best_gt[2]), int(best_gt[3]), int(best_gt[4]), int(best_gt[5])
                    cls_pred = classify_region(im[y1:y2, x1:x2])
                    if cls_pred == cls_true:
                        TP += 1
                    else:
                        FP += 1
                    matched.add(best_key)
                else:
                    # this GT was already matched
                    FP += 1
        FN += len([g for g in GTt if g[0]==n]) - len(matched)
    precision = TP/(TP+FP) if TP+FP else 0
    recall    = TP/(TP+FN) if TP+FN else 0
    f1_c      = 2*precision*recall/(precision+recall) if precision+recall else 0
    print(f"Classification Test → Acc:{precision:.3f}, R:{recall:.3f}, F1:{f1_c:.3f}")

# --- Evaluation on Learn set without running Test ---
def validate_learn():
    # Load GT for Learn
    GT_learn = sio.loadmat('progsPython/Apprentissage.mat')['BD']
    learn_ids = np.arange(1,262)[np.arange(1,262)%3==0]

    # 1) Segmentation-only evaluation on Learn
    seg_preds = []
    seg_gts   = []
    for n in learn_ids:
        im = cv2.imread(f"BD_METRO/IM ({n}).JPG")[...,::-1] / 255.0
        preds = segment_regions(im)
        gts   = [(int(g[3]), int(g[1]), int(g[4]), int(g[2])) for g in GT_learn if g[0]==n]
        seg_preds.extend(preds)
        seg_gts.extend(gts)
    p_s, r_s, f_s = evaluate_f1(seg_preds, seg_gts)
    print(f"Segmentation Learn → P:{p_s:.3f}, R:{r_s:.3f}, F1:{f_s:.3f}")

    # # 2) Classification evaluation via cross-validation on HOG features
    # X = np.array(hog_features)
    # y = np.array(hog_labels)
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # svm_cv = LinearSVC(max_iter=10000)
    # scores = cross_val_score(svm_cv, X, y, cv=cv, scoring='f1_macro')
    # print(f"Classification Learn (5-fold CV F1_macro): mean={scores.mean():.3f}, std={scores.std():.3f}")

    # # 3) Classification report on full Learn set
    # svm_full = LinearSVC(max_iter=10000).fit(X, y)
    # y_pred = svm_full.predict(X)
    # print("Classification report on full Learn set:")
    # print(classification_report(y, y_pred))
    
# --- Segmentation + classification pipeline ---

def segment_regions(image):
    # Preprocessing: blur to reduce noise
    img_uint8 = (image*255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (9,9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    # Multiple color ranges for metro line circles
    color_ranges = [
        ((20,100,100),(35,255,255)),    # M1: yellow
        ((100,150,50),(140,255,255)),   # M2: blue
        ((25,50,50),(45,255,200)),      # M3: olive
        ((85,50,50),(95,255,255)),      # M3bis: cyan
        ((140,100,100),(170,255,255)),  # M4: magenta/pink
        ((10,150,150),(25,255,255)),    # M5: orange
        ((40,50,50),(70,255,200)),      # M6: light green
        ((140,100,100),(170,255,255)),  # M7: rose (reuse pink)
        ((40,50,50),(70,255,200)),      # M7bis: mint green (reuse light green)
        ((130,50,50),(160,255,255)),    # M8: lavender
        ((15,100,100),(30,255,255)),    # M9: mustard yellow
        ((15,100,100),(30,255,255)),    # M10: ocher (reuse mustard)
        ((10,50,50),(20,255,200)),      # M11: brown
        ((35,100,100),(85,255,255)),    # M12: green
        ((80,50,100),(100,255,255)),    # M13: turquoise
        ((130,100,100),(160,255,255)),  # M14: purple
    ]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in color_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    boxes = []
    # 1) Contours + circularity
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(peri*peri) if peri>0 else 0
        if area >= area_min and circ >= circ_thresh:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h>0 else 0
            # filter nearly square regions only
            if 0.9 <= aspect <= 1.1:
                boxes.append((x, y, x+w, y+h))
    # 2) HoughCircles fallback
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)[:,:,0]
    circles = cv2.HoughCircles(gray_mask, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=30, param1=50, param2=20,
                               minRadius=10, maxRadius=100)
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            x1, y1 = max(x-r,0), max(y-r,0)
            x2, y2 = x1 + 2*r, y1 + 2*r
            boxes.append((x1, y1, x2, y2))
    return boxes

# Classification via HOG+SVM

def classify_region(crop):
    if not os.path.exists(svm_model_path):
        raise FileNotFoundError(f"SVM model not found: {svm_model_path}. Run Learn first.")
    svm = joblib.load(svm_model_path)
    c8  = cv2.resize((crop*255).astype(np.uint8),(64,64))
    gray= cv2.cvtColor(c8,cv2.COLOR_RGB2GRAY)
    feats = hog(gray,orientations=9,pixels_per_cell=(8,8),
                cells_per_block=(2,2),feature_vector=True)
    return int(svm.predict([feats])[0])

# Main pipeline

def metro2025(mode='Learn', viewImages=1):
    plt.close('all')
    print(f"[metro2025] Starting in mode {mode}, viewImages={viewImages}")
    ids = np.arange(1,262)
    if mode=='Learn':
        image_ids = ids[ids%3==0]
        GT = sio.loadmat('progsPython/Apprentissage.mat')['BD']
        hog_features.clear(); hog_labels.clear()
    else:
        image_ids = ids[ids%3!=0]
    print(f"[metro2025] {len(image_ids)} images to process in {mode}")

    # load reference BD for overlay
    refBD = None
    if viewImages:
        if mode == 'Learn':
            refBD = sio.loadmat('progsPython/Apprentissage.mat')['BD']
        else:
            refBD = sio.loadmat('progsPython/X.mat')['BD']

    BD = []
    for n in image_ids:
        print(f"[metro2025] Image {n}")
        im = np.array(Image.open(f"BD_METRO/IM ({n}).JPG").convert('RGB'))/255.0
        # if viewImages:
        #     if mode == 'Test':
        #         # Side-by-side: reference vs detections
        #         boxes_det = segment_regions(im)
        #         print(f"Image {n} → {len(boxes_det)} ROI détectées")
        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        #         # Left: reference (X.mat)
        #         ax1.imshow(im)
        #         ax1.set_title('Reference (blue)')
        #         idxs = np.where(refBD[:,0] == n)[0]
        #         for i in idxs:
        #             b = np.round(refBD[i,1:5]).astype(int)
        #             ax1.add_patch(Rectangle((b[2], b[0]), b[3]-b[2], b[1]-b[0], linewidth=2, edgecolor='b', facecolor='none'))
        #         # Right: detections (magenta)
        #         ax2.imshow(im)
        #         ax2.set_title('Detections (magenta)')
        #         for (x1, y1, x2, y2) in boxes_det:
        #             ax2.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='m', facecolor='none'))
        #         plt.tight_layout()
        #         plt.show()
        #         plt.close(fig)
            # else:
            #     # Learn: show GT boxes only in green
            #     plt.figure(figsize=(8,8))
            #     plt.imshow(im)
            #     idxs = np.where(refBD[:,0] == n)[0]
            #     for i in idxs:
            #         b = np.round(refBD[i,1:5]).astype(int)
            #         draw_rectangle(b[2], b[3], b[0], b[1], 'g')
            #     plt.title(f"GT Learn boxes (green)")
            #     plt.show()
            #     plt.close()

        if mode=='Learn':
            # extract HOG from GT boxes
            inds = np.where(GT[:,0]==n)[0]
            for k in inds:
                b = np.round(GT[k,1:5]).astype(int)
                crop = im[b[0]:b[1], b[2]:b[3]]
                crop8 = cv2.resize((crop*255).astype(np.uint8),(64,64))
                gray = cv2.cvtColor(crop8,cv2.COLOR_RGB2GRAY)
                feats = hog(gray, orientations=9,
                            pixels_per_cell=(8,8),cells_per_block=(2,2), feature_vector=True)
                hog_features.append(feats)
                hog_labels.append(int(GT[k,5]))
            # negative samples
            h,w = im.shape[:2]
            gt_boxes = [(int(GT[k,3]),int(GT[k,1]),int(GT[k,4]),int(GT[k,2])) for k in inds]
            for _ in range(len(inds)*neg_samples_per_gt):
                x0,y0 = np.random.randint(0,w-64), np.random.randint(0,h-64)
                boxN = (x0,y0,x0+64,y0+64)
                if all(iou(boxN,gb)<0.1 for gb in gt_boxes):
                    cropN = im[y0:y0+64,x0:x0+64]
                    crop8 = cv2.resize((cropN*255).astype(np.uint8),(64,64))
                    grayN = cv2.cvtColor(crop8,cv2.COLOR_RGB2GRAY)
                    featsN = hog(grayN,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)
                    hog_features.append(featsN); hog_labels.append(0)

        # detection/classification
        bd_list = []
        boxes = segment_regions(im) if mode=='Test' else []
        if mode=='Test':
            for (x1,y1,x2,y2) in boxes:
                cls = classify_region(im[y1:y2, x1:x2])
                bd_list.append([n,y1,y2,x1,x2,cls])
        else:
            for k in np.where(GT[:,0]==n)[0]:
                b = np.round(GT[k,1:5]).astype(int)
                bd_list.append([n,b[0],b[1],b[2],b[3],int(GT[k,5])])
        BD.extend(bd_list)

    if mode=='Learn':
        svm = LinearSVC(max_iter=10000)
        svm.fit(hog_features,hog_labels)
        joblib.dump(svm,svm_model_path)
        print(f"Trained SVM on {len(hog_features)} samples saved to {svm_model_path}")

    # save results
    sio.savemat('myResults.mat',{'BD':np.array(BD,dtype=int)})
    print(f"Saved myResults.mat for {mode}")

    # evaluation in Test
    if mode=='Test':
        evaluate_full_test()
        

if __name__=='__main__':
    # Run Learn then Test
    #metro2025('Learn', viewImages=1)
    #validate_learn()
    metro2025('Test', viewImages=1)
