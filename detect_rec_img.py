import argparse
import torch
import numpy as np
from utils import cfg_mnet, py_cpu_nms, decode, decode_landm, PriorBox
import cv2
from PIL import Image, ImageDraw, ImageFont
from net import Retina
import time
from rec import plate_recogition
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

class license_plate_ocr:
    def __init__(self, confidence_threshold=0.02, top_k=1000, nms_threshold=0.4, keep_top_k=500, vis_thres=0.6):
        self.net = Retina(cfg=cfg_mnet).to(device)
        self.net = load_model(self.net, 'mnet_plate.pth', False)
        self.net.eval()
        self.lprnet = plate_recogition()
        self.priorbox = PriorBox(cfg_mnet)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.resize = 1
        self.points_ref = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
    def det_rec(self, srcimg):
        img = np.float32(srcimg)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        with torch.no_grad():
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            loc, conf, landms = self.net(img)  # forward pass
            prior_data = self.priorbox((im_height, im_width)).to(device)

            boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
            boxes = boxes * scale / self.resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
            landms = landms * scale.repeat(2) / self.resize
            landms = landms.cpu().numpy()

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, self.nms_threshold,force_cpu=self.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        # show image
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            # text = "{:.4f}".format(b[4])
            # print(text)
            b = list(map(int, b))

            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            img_box = srcimg[y1:y2 + 1, x1:x2 + 1, :]
            new_x1, new_y1 = b[9] - x1, b[10] - y1
            new_x2, new_y2 = b[11] - x1, b[12] - y1
            new_x3, new_y3 = b[7] - x1, b[8] - y1
            new_x4, new_y4 = b[5] - x1, b[6] - y1

            # 定义对应的点
            points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
            # 计算得到转换矩阵
            M = cv2.getPerspectiveTransform(points1, self.points_ref)
            # 实现透视变换转换
            processed = cv2.warpPerspective(img_box, M, (94, 24))
            result = self.lprnet.rec(processed)

            cv2.rectangle(srcimg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # landms
            cv2.circle(srcimg, (b[5], b[6]), 2, (255, 0, 0), thickness=5)
            cv2.circle(srcimg, (b[7], b[8]), 2, (255, 0, 0), thickness=5)

            cv2.circle(srcimg, (b[9], b[10]), 2, (255, 0, 0), thickness=5)
            cv2.circle(srcimg, (b[11], b[12]), 2, (255, 0, 0), thickness=5)
            # cv2.putText(srcimg, result, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=3)
            srcimg = puttext_chinese(srcimg, result, (b[0], b[1] - 30), (0, 255, 0))
        return srcimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaPL')
    parser.add_argument('--imgpath', default='imgs/3.jpg', type=str, help='show detection results')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    model = license_plate_ocr(confidence_threshold=args.confidence_threshold, top_k=args.top_k, nms_threshold=args.nms_threshold, keep_top_k=args.keep_top_k, vis_thres=args.vis_thres)
    srcimg = cv2.imread(args.imgpath)
    srcimg = model.det_rec(srcimg)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', srcimg)
    if cv2.waitKey(1000000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
