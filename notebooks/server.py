import sys
sys.path.append('../')

from fastapi import FastAPI, File, UploadFile
import io
import cv2
import numpy as np
from starlette.responses import StreamingResponse

object_detection_server = FastAPI()


@object_detection_server.post('/detect_object')
def detect_object(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Save array as image
    cv2.imwrite('uploaded_img.jpg', my_img)

    return StreamingResponse(io.BytesIO(my_img.tobytes()), media_type="image/jpg")

