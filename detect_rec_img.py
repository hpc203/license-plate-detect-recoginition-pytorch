import argparse
import torch
import numpy as np
from utils import cfg_mnet, py_cpu_nms, decode, decode_landm, PriorBox
import cv2
from PIL import Image, ImageDraw, ImageFont
from net import Retina
import time
from rec import plate_recogition

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)  ###[:,:,::-1]  BGRtoRGB
    draw = ImageDraw.Draw(pilimg)  # 图片上打印汉字
    fontsize = int(min(img.shape[:2]) * 0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    draw.text(point, text, color, font=font)
    img = np.asarray(pilimg)  ###[:,:,::-1]   BGRtoRGB
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaPL')
    parser.add_argument('--imgpath', default='imgs/3.jpg', type=str, help='show detection results')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()
    cfg = cfg_mnet

    # net and model
    net = Retina(cfg=cfg)
    net = load_model(net, 'mnet_plate.pth', False)
    net.eval()

    lprnet = plate_recogition()
    print('Finished loading model!')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    resize = 1
    img_raw = cv2.imread(args.imgpath, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    print('priorBox time: {:.4f}'.format(time.time() - tic))
    # show image
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        # text = "{:.4f}".format(b[4])
        # print(text)
        b = list(map(int, b))

        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        img_box = img_raw[y1:y2 + 1, x1:x2 + 1, :]
        new_x1, new_y1 = b[9] - x1, b[10] - y1
        new_x2, new_y2 = b[11] - x1, b[12] - y1
        new_x3, new_y3 = b[7] - x1, b[8] - y1
        new_x4, new_y4 = b[5] - x1, b[6] - y1

        # 定义对应的点
        points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
        points2 = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
        # 计算得到转换矩阵
        M = cv2.getPerspectiveTransform(points1, points2)

        #实现透视变换转换
        processed = cv2.warpPerspective(img_box, M, (94, 24))
        # 显示原图和处理后的图像
        result = lprnet.rec(processed)
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        # landms
        cv2.circle(img_raw, (b[5], b[6]), 2, (255, 0, 0), thickness=5)
        cv2.circle(img_raw, (b[7], b[8]), 2, (255, 0, 0), thickness=5)

        cv2.circle(img_raw, (b[9], b[10]), 2, (255, 0, 0), thickness=5)
        cv2.circle(img_raw, (b[11], b[12]), 2, (255, 0, 0), thickness=5)
        # cv2.putText(img_raw, result, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=3)
        img_raw = puttext_chinese(img_raw, result, (b[0], b[1]-30), (0, 255, 0))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_raw)
    if cv2.waitKey(1000000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()