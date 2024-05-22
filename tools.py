# File with some tools

# imports
import cv2


# ---------------------------------------
#           Basic tools:
# ---------------------------------------
def rec_to_cor(x1, y1, x2, y2):
    """
    The function gets two points of rectangle (x1,y1) and (x2,y2), and return it as (x1,y1,h,w).
    [For openCV crops].
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: tuple
    """

    return [x1, y1, abs(x2 - x1), y2 - y1]


print(rec_to_cor(20, 30, 10, 10))


# ---------------------------------------
#           Image tools:
# ---------------------------------------
def many_crops(filename, rects, output):
    """
    The function gets an image file and list of rectangles, crop all the rectangles aand save
    them as "output+i.jpg"
    :param filename:
    :param rects:
    :param output:
    :return:
    """
    # Load the image
    image = cv2.imread(filename)
    # List of coordinates for the rectangles (x, y, width, height)
    # rectangles = [
    #     (50, 50, 100, 150),  # (x, y, width, height)
    #     (200, 80, 120, 100),
    #     (300, 200, 150, 150)
    # ]

    # Iterate through each rectangle and crop the image
    for i, (x, y, w, h) in enumerate(rects):
        # Crop the image
        cropped_image = image[y:y + h, x:x + w]

        # Save the cropped image with a unique filename
        filename = f'output{i + 1}.jpg'
        cv2.imwrite(filename, cropped_image)
        print(f'Cropped image saved as {filename}')


def find_let(img,output):
    """

    :param img:
    :param filename:
    :return:
    """
    image = cv2.imread(img)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test",img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1500]

    # draw contours on the original image
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=filtered_contours, contourIdx=-1, color=(0, 255, 0), thickness=1,lineType=cv2.LINE_AA)

    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('contours_none_image1.jpg', image_copy)
    cv2.destroyAllWindows()

    return filtered_contours


c = find_let("test1.jpg","test")
# Iterate through each contour
image = cv2.imread("test1.jpg")
for i, contour in enumerate(c):
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    print(x,y,w,h)
    # Crop the image using the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]
    # Save the cropped image
    cv2.imwrite(f'test1-output{i+1}.jpg', cropped_image)

