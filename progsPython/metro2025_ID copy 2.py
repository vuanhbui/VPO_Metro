# metro2025_full.py
import os
import numpy as np
import scipy.io as sio
import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# --- Pipeline parameters ---
area_min         = 3000    # segmentation minimal area
circ_thresh      = 0.85    # minimal circularity
neg_samples_per_gt = 2     # negative samples per GT in Learn
svm_model_path   = 'svm_model.pkl'

# --- Helpers ---
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

# --- Segmentation + classification pipeline ---
def segment_regions(im):
    hsv = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    m1  = cv2.inRange(hsv,(0,100,100),(15,255,255))
    m2  = cv2.inRange(hsv,(165,100,100),(180,255,255))
    mask= cv2.bitwise_or(m1,m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask   = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    cnts,_= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area    = cv2.contourArea(c)
        peri    = cv2.arcLength(c,True)
        circ    = 4*np.pi*area/(peri*peri) if peri>0 else 0
        if w*h>=area_min and circ>=circ_thresh:
            boxes.append((x,y,x+w,y+h))
    return boxes

def classify_region(crop):
    if not os.path.exists(svm_model_path):
        raise FileNotFoundError(f"SVM model not found: {svm_model_path}. Run Learn first.")
    svm = joblib.load(svm_model_path)
    c8  = cv2.resize((crop*255).astype(np.uint8),(64,64))
    gray= cv2.cvtColor(c8,cv2.COLOR_RGB2GRAY)
    feats = hog(gray,orientations=9,pixels_per_cell=(8,8),
                cells_per_block=(2,2),feature_vector=True)
    return int(svm.predict([feats])[0])

def metro2025(mode='Learn', viewImages=1):
    GT = sio.loadmat('progsPython/Apprentissage.mat')['BD'] if mode=='Learn' else None
    svm_features=[]; svm_labels=[]
    ids = np.arange(1,262)
    image_ids = ids[ids%3==0] if mode=='Learn' else ids[ids%3!=0]
    for n in image_ids:
        im = np.array(Image.open(f"BD_METRO/IM ({n}).JPG").convert('RGB'))/255.0
        if mode=='Learn':
            # positives
            inds= np.where(GT[:,0]==n)[0]
            for k in inds:
                b= np.round(GT[k,1:5]).astype(int)
                crop= im[b[0]:b[1],b[2]:b[3]]
                feats = hog(cv2.cvtColor(cv2.resize((crop*255).astype(np.uint8),(64,64)),cv2.COLOR_RGB2GRAY),
                            orientations=9,pixels_per_cell=(8,8),
                            cells_per_block=(2,2),feature_vector=True)
                svm_features.append(feats); svm_labels.append(int(GT[k,5]))
            # negatives
            h,w = im.shape[:2]
            gt_b = [ (int(GT[k,3]),int(GT[k,1]),
                      int(GT[k,4]),int(GT[k,2])) for k in inds ]
            for _ in range(len(inds)*neg_samples_per_gt):
                x0,y0 = np.random.randint(0,w-64), np.random.randint(0,h-64)
                box  =(x0,y0,x0+64,y0+64)
                if all(iou(box,gb)<0.1 for gb in gt_b):
                    cN=cv2.cvtColor(cv2.resize((im[y0:y0+64,x0:x0+64]*255).astype(np.uint8),(64,64)),cv2.COLOR_RGB2GRAY)
                    featsN= hog(cN,orientations=9,pixels_per_cell=(8,8),
                                cells_per_block=(2,2),feature_vector=True)
                    svm_features.append(featsN); svm_labels.append(0)
        # display (optional)
        if viewImages:
            plt.figure(figsize=(6,6)); plt.imshow(im)
            if mode=='Learn': # show GT
                for k in np.where(GT[:,0]==n)[0]:
                    b=np.round(GT[k,1:5]).astype(int)
                    draw_rectangle(b[2],b[3],b[0],b[1],'g')
                plt.title(f"GT Learn image {n}")
            plt.show(); plt.close()
    # train SVM
    if mode=='Learn':
        grid = GridSearchCV(LinearSVC(max_iter=10000), {'C':[0.01,0.1,1,10]},
                            scoring='f1_macro', cv=5)
        grid.fit(svm_features,svm_labels)
        best = grid.best_estimator_
        joblib.dump(best, svm_model_path)
        print(f"SVM trained, best C={grid.best_params_['C']}, F1={grid.best_score_:.3f}")
    else:
        # save results & evaluate overall
        evaluate_full_test()

def evaluate_full_test():
    GTt= sio.loadmat('progsPython/Test.mat')['BD']
    test_ids = np.arange(1,262)[np.arange(1,262)%3!=0]
    # segmentation only
    all_preds=[]; all_gts=[]
    for n in test_ids:
        im=cv2.imread(f"BD_METRO/IM ({n}).JPG")[...,::-1]/255.0
        preds = segment_regions(im)
        gts = [(int(g[3]),int(g[1]),int(g[4]),int(g[2])) 
               for g in GTt if g[0]==n]
        all_preds.extend(preds); all_gts.extend(gts)
    p_s,r_s,f_s = evaluate_f1(all_preds,all_gts)
    print(f"Segmentation Test → P:{p_s:.3f}, R:{r_s:.3f}, F1:{f_s:.3f}")
    # classification
    TP=FP=FN=0
    for n in test_ids:
        im=cv2.imread(f"BD_METRO/IM ({n}).JPG")[...,::-1]/255.0
        preds = segment_regions(im)
        gts = [g for g in GTt if g[0]==n]
        matched=set()
        for pb in preds:
            best_i=0; best_gt=None
            for g in gts:
                gb=(int(g[3]),int(g[1]),int(g[4]),int(g[2]))
                val=iou(pb,gb)
                if val>best_i: best_i,best_gt=val,g
            if best_i>=0.5 and best_gt not in matched:
                cls_pred = classify_region(
                    im[best_gt[1]:best_gt[2], best_gt[3]:best_gt[4]])
                cls_true = int(best_gt[5])
                if cls_pred==cls_true: TP+=1
                else: FP+=1
                matched.add(best_gt)
        FN += len(gts)-len(matched)
    acc = TP/(TP+FP) if TP+FP else 0
    rec = TP/(TP+FN) if TP+FN else 0
    f1c=2*acc*rec/(acc+rec) if acc+rec else 0
    print(f"Classification Test → Acc:{acc:.3f}, R:{rec:.3f}, F1:{f1c:.3f}")

# --- Main ---
if __name__=='__main__':
    # 1) Learn phase (with display)
    metro2025('Learn', viewImages=1)
    # 2) Test phase (no display)
    metro2025('Test', viewImages=0)