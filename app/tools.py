import numpy as np
import cv2


def draw_boxes(
    img : np.ndarray, rec_coordinates : np.ndarray, labels : list, colors : np.ndarray = None, relative_coordinates : bool = False,
    rec_thickness : int = 3, label_font_scale : int = 0.5, label_font_thickness : int = 1):
    """
    Method sued to draw boxes and add text to an image.

    Args:
        img (np.ndarray): Image contained in an numpy array with shape (img_height, img_width, channels). Where channels are BGR (3 channels)
        rec_coordinates (np.ndarray): Array containing the list of rectangles to draw. Each element on the array must contain the following elements:
        [X_start, Y_start, X_end, Y_end]. Where the coordinates are measured from top left of the image.
        labels (list): List of labels string to add to the image.
        colors (np.ndarray, optional): Array of colors to use for the boxes. Each element on the array must contain the following elemnts [B_value, G_value, R_value].
        If None is passed BOXES_COLORS will be used as default. Defaults to None.
        relative_coordinates (bool, optional): Bool value to convert \'rec_coordinates\' to relative coordinates. Defaults to False.
        rec_thickness (int, optional): Rectangles thickness. Defaults to 3.
        label_font_scale (int, optional): Font scale for text. Defaults to 0.5.
        label_font_thickness (int, optional): Text thickness. Defaults to 1.

    Returns:
        [type]: [description]
    """
    # rec_coordinates X_start, Y_start, X_end, Y_end (Measuring from top left)
    # img should be an array of shape (img_height, img_width, channels)
    # CV2 uses BGR
    BOXES_COLORS = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0],
          [1, 1, 0], [0, 1, 1], [1, 0, 1]]) * 255.0

    input_rec_coor = rec_coordinates.copy()
    input_img = img.copy()

    # Sanity check
    if not isinstance(labels, list):
        msg = '\'labels\' is not a list. Detected type: \'{}\''.format(type(labels))
        raise ValueError(msg)
    if not isinstance(input_img, np.ndarray):
        msg = '\'img\' is not a numpy array. Detected type: \'{}\''.format(type(input_img))
        raise ValueError(msg)
    if input_img.ndim != 3:
        msg = 'Input imgs must have 3 dimensions: (img_height, img_width, n_channels). You passed this shape: \'{}\''.format(input_imgs.shape)
        raise ValueError(msg)
    if not isinstance(input_rec_coor, np.ndarray):
        msg = 'Coordinates is not a numpy array. {}'.format(type(input_rec_coor))
        raise ValueError(msg)
    if (input_rec_coor.ndim != 2) and (input_rec_coor.shape[1] != 4):
        msg = 'Coordinates must have this shape [[y, x, y, x], ... ,[y, x, y, x]]. {}'.format(input_rec_coor)
        raise ValueError(msg)
    if len(labels) != input_rec_coor.shape[0]:
        msg = 'labels ({}) and rec_coordinates ({}) have different lengths.'.format(len(labels), input_rec_coor.shape[0])
        raise ValueError(msg)
    if relative_coordinates:
        img_width = input_img.shape[1]
        img_height = input_img.shape[0]
        input_rec_coor[:, [0, 2]] = input_rec_coor[:, [0, 2]] * img_width # img_height
        input_rec_coor[:, [1, 3]] = input_rec_coor[:, [1, 3]] * img_height # img_width
        input_rec_coor = np.asarray(input_rec_coor, dtype=int)

    # number of rectangles to draw
    n_elements = input_rec_coor.shape[0]
    
    # Single image passed to this function passed to the function
    v_img_size = input_img.shape[0]
    h_img_size = input_img.shape[1]
    channels_img = input_img.shape[2]
    
    # Draw rectabgles one by one
    for i in range(n_elements):
        # Draw rectangle on image
        start_point_i = tuple((input_rec_coor[i][:2]).astype(int))
        end_point_i = tuple((input_rec_coor[i][2:]).astype(int))
        
        color_i = tuple(BOXES_COLORS[i % BOXES_COLORS.shape[0]])
        #print('Drawing rectangle on coordinates: {}'.format(tuple(input_rec_coor[i])))
        #print('Using color: {}'.format(color_i))
        
        # Draw rectangle
        input_img = cv2.rectangle(
            img=input_img,
            pt1=start_point_i, 
            pt2=end_point_i,
            color=color_i,
            thickness=rec_thickness)
        # Get text size
        text_size = cv2.getTextSize(
            text=labels[i],
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=label_font_scale,
            thickness=label_font_thickness)[0]
        v_text_size = text_size[1]
        h_text_size = text_size[0]
        # Draw text background rectangle
        input_img = cv2.rectangle(
            img=input_img,
            pt1=(start_point_i[0] - rec_thickness, start_point_i[1] - rec_thickness), 
            pt2=(start_point_i[0] - rec_thickness + h_text_size, start_point_i[1] - rec_thickness - v_text_size),
            color=color_i, # Black background
            thickness=-1)
        # Add Text
        input_img = cv2.putText(
            img=input_img,
            text=labels[i],
            org=(start_point_i[0] - rec_thickness, start_point_i[1] - rec_thickness),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=label_font_scale, 
            color=(0, 0, 0), # Black text
            thickness=label_font_thickness)

    return input_img