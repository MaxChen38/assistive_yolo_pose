import sys
sys.path.append('/home/user/openpose/build/python')

import cv2
from openpose import pyopenpose as op

params = {
    "model_folder": "/home/user/openpose/models"
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    datum = op.Datum()
    datum.cvInputData = frame
    datum_vector = op.VectorDatum()
    datum_vector.append(datum)
    opWrapper.emplaceAndPop(datum_vector)

    cv2.imshow("OpenPose", datum.cvOutputData)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
