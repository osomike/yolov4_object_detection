from fastapi import FastAPI, File, UploadFile
import io
import cv2
import numpy as np
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

# idea 1 https://www.youtube.com/watch?v=LHOjW42-A40&ab_channel=RedisLabs
# idea 2 https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
my_app = FastAPI()


@my_app.get("/")
async def main():
    """
    Main app. Return Hello World text
    """
    return {"message": "Hello World"}

@my_app.get("/test")
async def test():
    return {"message": "This is a test"}

@my_app.get("/increment/{int1}")
async def increment(int1: int):
    """
    Increment value by 1
    @params int1 : input value
    """
    int1 += 1
    return {"New value": int1}

@my_app.post("/files")
async def file_upload(
    avatar: UploadFile = File(...),
    banner: UploadFile = File(...)
):
    print(avatar)
    print(banner)
    open("fd.jpg","wb").write(avatar.file.read())
    return "top"


class Item(BaseModel):
    img_name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

class MyImage(BaseModel):
    img_name: str
    img_binary: bytes


@my_app.post("/items/")
async def create_item(item: Item):
    return item.img_name

@my_app.post("/img/")
async def create_item(img: bytes):
    
    return type(img)

@my_app.post('/img_file')
def _file_upload(my_file: UploadFile = File(...)):
    print('Received!')
    print(my_file.file.read())
    return 1

@my_app.post('/send_img')
def _file_upload(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)
    print('my_array: {}'.format(my_array[:10]))

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Save array as image
    cv2.imwrite('uploaded_img.jpg', my_img)
    print('tobytes: {}'.format(my_img.tobytes()[:10]))
    #print('BytesIO: {}'.format(io.BytesIO(my_img)[:10]))
    return StreamingResponse(io.BytesIO(my_img.tobytes()), media_type="image/jpg")
    #return StreamingResponse(io.BytesIO(my_img), media_type="image/jpg")


@my_app.post("/vector_image")
def image_endpoint(*, vector):
    # Returns a cv2 image array from the document vector
    cv2img = my_function(vector)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@my_app.post("/image_detection")
async def image_decoder(*, binary_file):
    content = await binary_file.read()
    print(type(content))
    return {'type': type(binary_file)}
    # convert string of image data to uint8
    #nparr = np.fromstring(binary_file.file, np.uint8)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    #return {'image shape': img.shape}