############## replace label text in list #################
def draw_boxes_texts(im, boxes, labels=None, color=None):
    boxes = np.asarray(boxes, dtype='int32')
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)    # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
        and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
        "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for idx, i in enumerate(sorted_inds, 1):
        box = boxes[i, :]
        if labels is not None:
            im = draw_text(im, (box[0], box[1]), str(idx), color=color) # draw label idx along rectangle
            im = draw_text(im, (10, idx*12+10), str(idx), color=color) # draw label idx top left
            im = draw_text(im, (30, idx*12+10), labels[i], color=color) # label text list from image top left
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                      color=color, thickness=1)

    # add red rectangle arround picture that with failure
    height, width, channels = im.shape
    cv2.rectangle(im, (0, 0), (width, height), color=(0, 0, 255), thickness=5)
    return im

############## replace label text in list #################
