import numpy, cv2

points = []
points.append(numpy.array([(143,1557),(198,1790),(1127,1501),(1248,1310)]))
points.append(numpy.array([(665,488),(739,713),(1632,391),(1736,193)]))
points.append(numpy.array([(889,1534),(1009,1861),(2344,1518),(2219,765)]))
points.append(numpy.array([(261,661),(224,998),(1560,1295),(1793,578)]))

img0 = cv2.imread("img/Foto424.jpg",cv2.IMREAD_COLOR)
img1 = cv2.imread("img/Foto425.jpg",cv2.IMREAD_COLOR)
img2 = cv2.imread("img/Foto426.jpg",cv2.IMREAD_COLOR)

warp0 = cv2.warpPerspective(img0, numpy.eye(3), (4000, 4000))
warp1 = cv2.warpPerspective(img1, cv2.findHomography(points[1],points[0])[0], (4000, 4000))

warp2 = cv2.warpPerspective(img2, cv2.findHomography(points[3],points[2])[0], (4000, 4000))
warp2 = cv2.warpPerspective(warp2, cv2.findHomography(points[1],points[0])[0], (4000, 4000))

cv2.namedWindow("CC",cv2.WINDOW_FREERATIO)
maximg = numpy.maximum(warp0,warp1)
maximg = numpy.maximum(maximg,warp2)

cv2.imwrite("immagine.jpg",maximg)