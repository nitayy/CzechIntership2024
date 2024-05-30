# File with some tools

# imports
import cv2


# ---------------------------------------
#           Basic tools:
# ---------------------------------------


# ---------------------------------------
#           Image tools:
# ---------------------------------------

def find_let(img, low_t=130, show=True):
    """

    :param show:
    :param low_t:
    :param img:
    :return:
    """
    path = "./tests/" + img + ".jpg"
    path_org = "./tests/" + img + "-org.jpg"
    image = cv2.imread(path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show:
        cv2.imshow("img", img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, low_t, 225, cv2.THRESH_BINARY)
    # # Apply Gaussian Blur
    # blurred_image = cv2.GaussianBlur(image, (1, 1), 0)
    #
    # # Perform Canny edge detection
    # edges = cv2.Canny(blurred_image, 100, 200)

    # Display the result using OpenCV
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # visualize the binary image
    if show:
        cv2.imshow('Binary image', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('image_thres1.jpg', thresh)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # draw contours on the original image
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=filtered_contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                     lineType=cv2.LINE_AA)

    # see the results
    if show:
        cv2.imshow('None approximation', image_copy)
        cv2.waitKey(0)
        cv2.imwrite('contours_none_image1.jpg', image_copy)
        cv2.destroyAllWindows()

    # Iterate through each contour and saving the output at "dest"
    image = cv2.imread(path_org)
    for i, contour in enumerate(filtered_contours):
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        # Crop the image using the bounding rectangle
        cropped_image = image[y:y + h, x:x + w]
        # Save the cropped image
        cv2.imwrite(f'./tests/{img}-output{i + 1}.jpg', cropped_image)

    return filtered_contours


c = find_let("test8", 130, True)
print(len(c))
