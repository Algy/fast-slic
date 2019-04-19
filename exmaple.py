from collections import Counter
N = 1600
slic = Slic(N, compactness=3)
while True:
    flag, frame = cap.read()
    if not flag:
        break
    assignment = slic.iterate(frame, 10)
    clusters = slic.slic_model.clusters
    cluster_colors = np.array([d['color'] for d in clusters], dtype=np.uint8)
    hist = cluster_colors[assignment]
    mask = model.predict(frame[:, :, ::-1]) >= 230

    classes = np.zeros([N], np.int32)
    for no, num in Counter(assignment[mask]).items():
        density = num / clusters[no]["num_members"]
        if density >= .5:
            classes[no] = 1
    crf = SimpleCRF(2, N)
    crf.spatial_w = 1
    crf.spatial_sxy = 90
    crf.spatial_srgb = 40
    crf.spatial_smooth_w = 3

    crf_frame = crf.push_slic_frame(slic, knn=30)
    crf_frame.set_mask(classes, 0.8)
    crf.initialize()
    crf.inference(10)
    if np.isnan(crf.get_frame(0).get_inferred()).any():
        raise RuntimeError("SADFDSF")
    t = (crf.get_frame(0).get_inferred()[1] >= 0.1).nonzero()[0]
    res = frame[:,:, ::-1].copy()
    '''
    res = frame[:,:, ::-1]
    t = classes.nonzero()[0]
    '''
    res[~np.isin(assignment, t)] = 0
    vis = res

    col = frame.copy()[:,:,::-1]
    col[~mask] = 0


    cv2.imshow("Show", vis[:,:,::-1])
    cv2.imshow("HIST", hist)
    cv2.imshow("Mask", col[:,:,::-1])
    cv2.waitKey(1)

