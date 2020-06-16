import cv2
import numpy as np
import time
from skimage import measure
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    print(hex_colors)
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

def hexer():
  get_colors(get_image('AH.jpg'), 1, True)


upper_left = (445, 306)
bottom_right = (525, 405)
flag = 1
counter_shareef = 0
cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

calc_timestamps = [0.0]

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
start=0
end=0
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 255), 5)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=6)

    

# RESEARCH WONDERFUL LINK : https://docs.opencv.org/3.3.1/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    
    if cv2.countNonZero(thresh) and flag !=0:
      start = time.time()
      counter_shareef += 1
      timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
      calc_timestamps.append(calc_timestamps[-1] + 1000/fps)

      cv2.imwrite('AH.jpg', rect_img)
      hexer()
      
      flag = 0
    else:
      flag = 1
      end = time.time()

    elapsed = end - start
    if elapsed < 0:
        print("The time for blink is: ", abs(elapsed*1000), "MS")


    cv2.putText(frame, "COUNTER VALUE : {}".format(counter_shareef), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
    cv2.imshow('Frame',frame)
    cv2.imshow('Cutting',gray)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Threshold', thresh)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

b = []
cap.release()
for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
##  print('Blink Duration %d Number:'%i,abs(ts - cts))
    b.append(abs(ts - cts))
    if i > 0:
      ans = b[i] - b[i-1]
      print('Duration in between Blink %d Number: '%i, ans/1000, 'MS')
    else:
      b[i] = abs(ts - cts)
      print(b[i])





cv2.destroyAllWindows()
