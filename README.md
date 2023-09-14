# faceswap-flask
swap your face with anyone in real time\
front-end with flask

# working
 * The model extracts the user's face using mediapipe's face_mesh solution
 * Then the face_points are extracted
 * The same is done for the source image
 * The face points are subdivided into triangles and affine transformation is applied so that the new triangle overlies the previous one
 * The final masked image released in the output video

 * A simple login/signup portal in flask
 * Live video in second page with place for uploading the source image