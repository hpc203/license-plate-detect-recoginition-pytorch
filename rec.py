import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class my_lprnet(nn.Module):
    def __init__(self, class_num):
        super(my_lprnet, self).__init__()
        self.class_num = class_num
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU())
        self.stage2 = nn.Sequential(small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU())
        self.stage3 = nn.Sequential(small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU())
        self.stage4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU())
        self.container = nn.Conv2d(in_channels=448 + class_num, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out1 = self.stage1(x)
        out = F.max_pool2d(out1, 3, stride=(1, 1))
        out2 = self.stage2(out)

        out = F.max_pool2d(out2, 3, stride=(1, 2))
        out = F.max_pool2d(out.permute(0, 2, 3, 1).contiguous(), 1, stride=(1, 2))
        out = out.permute(0, 3, 1, 2).contiguous()

        out3 = self.stage3(out)

        out = F.max_pool2d(out3, 3, stride=(1, 2))
        out = F.max_pool2d(out.permute(0, 2, 3, 1).contiguous(), 1, stride=(1, 4))
        out = out.permute(0, 3, 1, 2).contiguous()
        out4 = self.stage4(out)

        out1 = F.avg_pool2d(out1, kernel_size=5, stride=5)
        f = torch.pow(out1, 2)
        f = torch.mean(f)
        out1 = torch.div(out1, f.item())

        out2 = F.avg_pool2d(out2, kernel_size=5, stride=5)
        f = torch.pow(out2, 2)
        f = torch.mean(f)
        out2 = torch.div(out2, f.item())

        out3 = F.avg_pool2d(out3, kernel_size=(4, 10), stride=(4, 2))
        f = torch.pow(out3, 2)
        f = torch.mean(f)
        out3 = torch.div(out3, f.item())

        f = torch.pow(out4, 2)
        f = torch.mean(f)
        out4 = torch.div(out4, f.item())

        logits = torch.cat((out1, out2, out3, out4), 1)
        logits = self.container(logits)
        logits = torch.mean(logits, dim=2)
        logits = logits.view(self.class_num, -1)
        return logits

class plate_recogition:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = (94, 24)
        self.lprnet = my_lprnet(len(CHARS))
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(torch.load('my_lprnet_model.pth', map_location=self.device))
        self.lprnet.eval()
    def rec(self, img):
        height, width, _ = img.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            img = cv2.resize(img, self.img_size)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            preb = self.lprnet(img)
            preb = preb.detach().cpu().numpy().squeeze()
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))

        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        return ''.join(list(map(lambda x: CHARS[x], no_repeat_blank_label)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--imgpath', default='processed2.jpg', type=str, help='the image path')
    args = parser.parse_args()

    lprnet = plate_recogition()
    img = cv2.imread(args.imgpath)
    srcimg = img.copy()
    result = lprnet.rec(img)
    print(result)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()