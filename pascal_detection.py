
import cv2
from fastai.vision import *
from fastai.vision.models.unet import _get_sfs_idxs, model_sizes, hook_outputs
from torchvision import models, transforms
import imutils


COLORS = np.random.uniform(0, 255, size=(21, 3))


def draw_rect(output, b:Collection[int],text=None, col=0):

    "Draw bounding box on image."

    cv2.rectangle(output, (b[0],b[1]), (b[2],b[3]),COLORS[col], 2)
    if text is not None:
        y = b[1] - 10 if b[1] - 10 > 10 else b[1] + 10
        cv2.putText(output, text, (b[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[col], 1)


def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."
    if flatten:
        acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]])) #Can't remember where those scales come from, but they help regularize
        centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
        sizes = anchors[...,2:] * torch.exp(acts[...,:2])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res


def create_grid(size):
    "Create a grid of a given `size`."
    H, W = size if is_tuple(size) else (size,size)
    grid = FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1,2)


def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:,:2] - boxes[:,2:]/2
    bot_right = boxes[:,:2] + boxes[:,2:]/2
    return torch.cat([top_left, bot_right], 1)

def unpad(tgt_bbox, tgt_clas, pad_idx=0):
    i = torch.min(torch.nonzero(tgt_clas-pad_idx))
    return tlbr2cthw(tgt_bbox[i:]), tgt_clas[i:]-1+pad_idx


def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:,:2] + boxes[:,2:])/2
    sizes = boxes[:,2:] - boxes[:,:2]
    return torch.cat([center, sizes], 1)


def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a,t,4), tgts.unsqueeze(0).expand(a,t,4)
    top_left_i = torch.max(ancs[...,:2], tgts[...,:2])
    bot_right_i = torch.min(ancs[...,2:], tgts[...,2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[...,0] * sizes[...,1]


def IoU_values(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    inter = intersection(anchors, targets)
    anc_sz, tgt_sz = anchors[:,2] * anchors[:,3], targets[:,2] * targets[:,3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter/(union+1e-8)


def create_anchors(sizes, ratios, scales, flatten=True):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1,2)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h,2/w])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,2)], 2)
        anchors.append(ancs.view(h,w,a,4))
    return torch.cat([anc.view(-1,4) for anc in anchors],0) if flatten else anchors

def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = LongTensor(list(range(len(idxs))))
    target[i1s[mask],idxs[mask]-1] = 1
    return target


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    matches = anchors.new(anchors.size(0)).zero_().long() - 2
    if targets.numel() == 0: return matches
    ious = IoU_values(anchors, targets)
    vals,idxs = torch.max(ious,1)
    matches[vals < bkg_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    #Overwrite matches with each target getting the anchor that has the max IoU.
    #vals,idxs = torch.max(ious,0)
    #If idxs contains repetition, this doesn't bug and only the last is considered.
    #matches[idxs] = targets.new_tensor(list(range(targets.size(0)))).long()
    return matches

def bbox_to_activ(bboxes, anchors, flatten=True):
    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        t_centers = (bboxes[...,:2] - anchors[...,:2]) / anchors[...,2:]
        t_sizes = torch.log(bboxes[...,2:] / anchors[...,2:] + 1e-8)
        return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res


class LateralUpsampleMerge(nn.Module):
    "Merge the features coming from the downsample path (in `hook`) with the upsample path."

    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)

    def forward(self, x):
        return self.conv_lat(self.hook.stored) + F.interpolate(x, self.hook.stored.shape[-2:], mode='nearest')


class RetinaNet(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"

    def __init__(self, encoder: nn.Module, n_classes, final_bias=0., chs=256, n_anchors=9, flatten=True):
        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, sfs_szs[idx][1], hook)
                                     for idx, hook in zip(sfs_idxs[-2:-4:-1], self.sfs[-2:-4:-1])])
        self.smoothers = nn.ModuleList([conv2d(chs, chs, 3, bias=True) for _ in range(3)])
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs)
        self.box_regressor = self._head_subnet(4, n_anchors, 0., chs=chs)

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        "Helper function to create one of the subnet for regression/classification."
        layers = [conv_layer(chs, chs, bias=True, norm_type=None) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)

    def _apply_transpose(self, func, p_states, n_classes):
        # Final result of the classifier/regressor is bs * (k * n_anchors) * h * w
        # We make it bs * h * w * n_anchors * k then flatten in bs * -1 * k so we can contenate
        # all the results in bs * anchors * k (the non flatten version is there for debugging only)
        if not self.flatten:
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [func(p).permute(0, 2, 3, 1).view(*sz, -1, n_classes) for p, sz in zip(p_states, sizes)]
        else:
            return torch.cat(
                [func(p).permute(0, 2, 3, 1).contiguous().view(p.size(0), -1, n_classes) for p in p_states], 1)

    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges: p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        return [self._apply_transpose(self.classifier, p_states, self.n_classes),
                self._apply_transpose(self.box_regressor, p_states, 4),
                [[p.size(2), p.size(3)] for p in p_states]]

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


class RetinaNetFocalLoss(nn.Module):

    def __init__(self, gamma: float = 2., alpha: float = 0.25, pad_idx: int = 0, scales: Collection[float] = None,
                 ratios: Collection[float] = None, reg_loss: LossFunction = F.smooth_l1_loss):
        super().__init__()
        self.gamma, self.alpha, self.pad_idx, self.reg_loss = gamma, alpha, pad_idx, reg_loss
        self.scales = ifnone(scales, [1, 2 ** (-1 / 3), 2 ** (-2 / 3)])
        self.ratios = ifnone(ratios, [1 / 2, 1, 2])

    def _change_anchors(self, sizes: Sizes) -> bool:
        if not hasattr(self, 'sizes'): return True
        for sz1, sz2 in zip(self.sizes, sizes):
            if sz1[0] != sz2[0] or sz1[1] != sz2[1]: return True
        return False

    def _create_anchors(self, sizes: Sizes, device: torch.device):
        self.sizes = sizes
        self.anchors = create_anchors(sizes, self.ratios, self.scales).to(device)

    def _unpad(self, bbox_tgt, clas_tgt):
        i = torch.min(torch.nonzero(clas_tgt - self.pad_idx))
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:] - 1 + self.pad_idx

    def _focal_loss(self, clas_pred, clas_tgt):
        encoded_tgt = encode_class(clas_tgt, clas_pred.size(1))
        ps = torch.sigmoid(clas_pred.detach())
        weights = encoded_tgt * (1 - ps) + (1 - encoded_tgt) * ps
        alphas = (1 - encoded_tgt) * self.alpha + encoded_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(clas_pred, encoded_tgt, weights, reduction='sum')
        return clas_loss

    def _one_loss(self, clas_pred, bbox_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        matches = match_anchors(self.anchors, bbox_tgt)
        bbox_mask = matches >= 0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask]))
        else:
            bb_loss = 0.
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        return bb_loss + self._focal_loss(clas_pred, clas_tgt) / torch.clamp(bbox_mask.sum(), min=1.)

    def forward(self, output, bbox_tgts, clas_tgts):
        clas_preds, bbox_preds, sizes = output
        if self._change_anchors(sizes): self._create_anchors(sizes, clas_preds.device)
        n_classes = clas_preds.size(2)
        return sum([self._one_loss(cp, bp, ct, bt)
                    for (cp, bp, ct, bt) in zip(clas_preds, bbox_preds, clas_tgts, bbox_tgts)]) / clas_tgts.size(0)


def nms(boxes, scores, thresh=0.3):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.LongTensor(range_of(scores))
    while len(scores) > 0:
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals < thresh
        if len(mask_keep.nonzero()) == 0: break
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return LongTensor(to_keep)


def process_output(output, detect_thresh=0.35):
    clas_pred,bbox_pred,sizes = output[0][0], output[1][0], output[2]
    anchors = create_anchors(sizes, ratios, scales).to(clas_pred.device)
    bbox_pred = activ_to_bbox(bbox_pred, anchors)
    clas_pred = torch.sigmoid(clas_pred)
    detect_mask = clas_pred.max(1)[0] > detect_thresh
    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
    if clas_pred.numel() == 0: return [],[],[]
    scores, preds = clas_pred.max(1)
    return bbox_pred, scores, preds


def show_preds(sz, bgr_img, output, detect_thresh=0.35, classes=None):

    bbox_pred, scores, preds = process_output(output, detect_thresh)
    if len(scores) != 0:
        to_keep = nms(bbox_pred, scores)
        bbox_pred, preds, scores = bbox_pred[to_keep].cuda(), preds[to_keep].cuda(), scores[to_keep].cuda()
        bbox_pred[:,:2] = bbox_pred[:,:2] * sz
        bbox_pred[:,2:] = bbox_pred[:,2:] * sz
        bbox_pred = bbox_pred.long()

    for bbox, c, scr in zip(bbox_pred, preds, scores):
        txt = str(c.item()) if classes is None else classes[c.item()+1]
        col=c.item()+1
        draw_rect(bgr_img,[bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}',col=col)


if __name__ == '__main__':
    ratios = [1 / 2, 1, 2]
    scales = [1, 2 ** (-1 / 3), 2 ** (-2 / 3)]
    classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
 'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.Normalize,
                                          ])
    defaults.device = torch.device('cuda')
    encoder = create_body(models.resnet50, cut=-2)
    model =  RetinaNet(encoder,21,final_bias=-4)
    state_dict = torch.load('stage2-256.pth')
    model.load_state_dict(state_dict['model'],strict=False)
    model = model.cuda()
    model.eval()
    photo = cv2.imread('cars.jpg')
    t_sz = torch.Tensor([photo.shape[:2]])[None].cuda().float()
    image = imutils.resize(photo,width=256,height=256)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = pil2tensor(image, dtype=np.float32).div_(255)


    with torch.no_grad():
        z = model(image.unsqueeze_(0).cuda())
        show_preds(t_sz, photo, z, detect_thresh=0.4, classes=classes)

    cv2.imshow('img',photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








  #test on webcam video stream

    # cap = cv2.VideoCapture(0)
    #
    # while(True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     frame2 = imutils.resize(frame, width=256, height=256)
    #     image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    #     image = pil2tensor(image, dtype=np.float32).div_(255)
    #     t_sz = torch.Tensor([frame.shape[:2]])[None].cuda().float()
    #
    #     # draw the prediction on the output image
    #     with torch.no_grad():
    #         z = model(image.unsqueeze_(0).cuda())
    #         show_preds(t_sz, frame, z, detect_thresh=0.6, classes=classes)
    #     # # Display the resulting frame
    #
    #     cv2.imshow('img', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
