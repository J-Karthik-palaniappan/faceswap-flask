import cv2
import mediapipe as mp
import numpy as np

def func(img2,swapimg):
    #=====READING=====
    if swapimg:
        img = cv2.imread(swapimg)
    else:
        return img2
    face_mask = np.zeros(img2.shape, dtype = img2.dtype)
    
    #=====IMPORING MODEL=====
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)
    
    #=====MODEL OUTPUT=====
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    values = results.multi_face_landmarks[0].landmark
    results2 = face_mesh.process(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    values2 = results2.multi_face_landmarks[0].landmark
    
    
    #=====CONVERTING TO ARRAY FORM=====
    face_points = np.zeros((len(values),2))
    for i,val in enumerate(values):
        face_points[i][0] = val.x
        face_points[i][1] = val.y
    face_points*=(img.shape[1],img.shape[0])
        
    face_points2 = np.zeros((len(values2),2))
    for i,val in enumerate(values2):
        face_points2[i][0] = val.x
        face_points2[i][1] = val.y
    face_points2*=(img2.shape[1],img2.shape[0])
    
    required_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                     285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                     387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                     178, 162, 54, 67, 10, 297, 284, 389]
    face_points = face_points[required_indices]
    face_points2 = face_points2[required_indices]
    
    points = np.array(face_points,np.int32)
    convexhull = cv2.convexHull(points)
    
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(face_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles,dtype = np.int32)
    
    indices=[]
    for t in triangles:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt3 = np.where((points == pt3).all(axis=1))
    
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indices.append(triangle)
        
    triangles2 = []
    for i in indices:
        tr1_pt1 = face_points2[i[0]][0]
        tr1_pt2 = face_points2[i[1]][0]
        tr1_pt3 = face_points2[i[2]][0]
        triangles2.append([tr1_pt1,tr1_pt2,tr1_pt3])
    
    triangles2 = np.array(triangles2,dtype=np.int32)
    triangles = np.reshape(triangles,triangles2.shape)
    
    for i in range(len(triangles)):
        tri1 = triangles[i]
        tri2 = triangles2[i]
        # Find bounding box. 
        r1 = cv2.boundingRect(tri1)
        r2 = cv2.boundingRect(tri2)
        #input cropping
        tri1Cropped = []
        tri2Cropped = []
        for i in range(0, 3):
            tri1Cropped.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
            tri2Cropped.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))
        img1Cropped = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        #affine transform
        warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
        img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]))
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
        img2Cropped = img2Cropped * mask

        img2_new_face_rect_area = face_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        img2
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img2Cropped = cv2.bitwise_and(img2Cropped, img2Cropped, mask=mask_triangles_designed)
        
        img2_new_face_rect_area = img2_new_face_rect_area + np.array(img2Cropped)
        face_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_new_face_rect_area
        
    points2 = np.array(face_points2,np.int32)
    convexhull2 = cv2.convexHull(points2)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = img2_head_noface + face_mask
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

    return seamlessclone

camera = None

def open_camera():
    global camera
    if not camera:
        camera = cv2.VideoCapture(0)
    
def gen_frames(swapimg):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                new_img = func(frame,swapimg)
            except:
                new_img = frame
                
            ret, buffer = cv2.imencode('.jpg', new_img)
            new_img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + new_img + b'\r\n')
            
def close_camera():
    global camera
    if camera:
        camera.release()
        camera = None

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while(True):
        ret, sample_img = camera.read()
        try:
            new_img=func(sample_img)
        except:
            new_img=sample_img
        cv2.imshow("musk",new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()