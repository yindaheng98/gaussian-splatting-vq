import json
import math
import os
import argparse
import torch
from typing import NamedTuple, List
from scene.camera_dataset import CameraPoseDataset, Pose
from gaussian_renderer import render, GaussianModel
from scene.gaussian_vq import VQGaussianModel, Attribute
from scene.gaussian_kmeans import KMeansGaussianModel
from arguments import PipelineParams
from scene.cameras import Camera as View
from utils.system_utils import searchForMaxIteration
from warping import MorphologyClose, error_erosion, fromJSON, is_occlusion, reconstrucion, projection
from predictions.base import Prediction
from predictions import prediction_dict
from predictions.Linear import LinearPredictionFoV
from utils.camera_utils import matrix_to_quaternion
import pandas as pd
from itertools import cycle
from restore import get_restoration, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--cameras-start", type=int, default=0)
parser.add_argument("--fovx", type=float, default=1.4773902348773813, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=1.2005465997792715, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)  # 用前15帧做预测
parser.add_argument("--prediction-stride", type=int, default=9)  # 本地缓存3帧
parser.add_argument("--prediction-length", type=int, default=3)  # 每个reference image给3帧用
parser.add_argument("--use-enlarged-in-mark-visible", action="store_true")
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--max-frame", type=int, default=10)
parser.add_argument("--codebooks", type=str, required=True)
parser.add_argument("--bandwidth", type=str, default="saved_data/bandwidth.csv")
parser.add_argument("--bandwidth-start", type=int, default=0)
parser.add_argument("--bandwidth-end", type=int, default=None)
parser.add_argument("--prediction", type=str, default="VAR")
parser.add_argument("--prediction-conf", type=str, required=True)
parser.add_argument("--prediction-load", type=str)
parser.add_argument("--fov-save", type=str, default="output/run/fov.txt")
parser.add_argument("--trace-save", type=str, default="output/run/trace.json")
parser.add_argument("--image-save", type=str, default="output/run")
parser.add_argument("--restore-save", type=str, required=True)


class Camera(NamedTuple):
    pose: Pose
    fovx: float
    fovy: float
    width: int
    height: int


def camera2view(camera: Camera):
    return View(colmap_id="", R=camera.pose.R.cpu().numpy(), T=camera.pose.T.cpu().numpy(),
                FoVx=camera.fovx, FoVy=camera.fovy,
                image=None, gt_alpha_mask=None,
                image_name="", uid="",
                data_device="cuda",
                image_width=camera.width, image_height=camera.height)


def predict_camera(prediction: Prediction, pose_history, prediction_stride, prediction_length, fovx: float, fovy: float, width: int, height: int):
    with torch.no_grad():
        pose_pred = prediction.predict(pose_history, prediction_stride, prediction_length)
        return pose_pred


def render_frame(camera: Camera, gaussians: GaussianModel, pipeline: PipelineParams):
    view = camera2view(camera)
    with torch.no_grad():
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        return rendering, depth


def warp(uv, color_ref, depth):
    height, width = color_ref.shape[:2]
    # done by grid_sample, same result, may be faster?
    # grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    # warped = F.grid_sample(color_ref.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
    #                        mode='bilinear', align_corners=True)[0, ...].type(torch.uint8).permute(1, 2, 0)
    uv_idx = uv[..., :2]
    uv_idx = uv_idx.round().type(torch.int64)
    is_edge = uv_idx[..., 1] < 0
    is_edge |= uv_idx[..., 1] >= height - 1
    is_edge |= uv_idx[..., 0] < 0
    is_edge |= uv_idx[..., 0] >= width - 1
    # w_enlarge = max(-uv_idx[..., 0].min(), uv_idx[..., 0].max() - width + 1)
    # h_enlarge = max(-uv_idx[..., 1].min(), uv_idx[..., 1].max() - height + 1)
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
    # warped = torch.zeros_like(color_ref)  # inverse
    # warped[uv_idx[..., 1], uv_idx[..., 0], ...] = color_ref  # inverse

    mask_occluded, mask_occlude = is_occlusion(uv_idx, depth, height, width)
    mask_occluded = MorphologyClose(mask_occluded)
    mask_occlude = MorphologyClose(mask_occlude)
    # warped[mask_occluded, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # return warped

    # mask_occluded_last = mask_occluded.clone()  # debug
    kernel_size, occluded_dilation_size, occlude_dilation_size = 8, 5, 5
    warped, mask_occluded, validcount = error_erosion(
        warped, mask_occluded, mask_occlude,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size)
    # print(validcount, mask_occluded.sum())  # debug
    while mask_occluded.sum() > 0 and validcount > 0:
        warped, mask_occluded, validcount = error_erosion(
            warped, mask_occluded, mask_occlude,
            kernel_size=kernel_size,
            occluded_dilation_size=occluded_dilation_size,
            occlude_dilation_size=occlude_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
        if validcount <= 0:
            occluded_dilation_size -= 1
            occlude_dilation_size -= 1
            warped, mask_occluded, validcount = error_erosion(
                warped, mask_occluded, mask_occlude,
                kernel_size=kernel_size,
                occluded_dilation_size=occluded_dilation_size,
                occlude_dilation_size=occlude_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
    # warped[mask_occluded_last, :] = torch.tensor([255, 0, 0], dtype=warped.dtype)  # debug
    # warped[mask_occluded, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[is_edge, ...] = 0
    return warped, is_edge


def warping_frame(camera: Camera, depth, camera_ref: Camera, color_ref):
    K, R_c2w, T_c2w, _, _ = fromJSON(camera2view(camera).toJSON(0))
    xyz = reconstrucion(K, R_c2w, T_c2w, depth)
    K_r, R_r, t_r, _, _ = fromJSON(camera2view(camera_ref).toJSON(0))
    uv, z = projection(K_r, R_r, t_r, xyz)
    warped, is_edge = warp(uv, color_ref.permute(1, 2, 0), z)  # wrap it
    return warped.permute(2, 0, 1), is_edge


def compute_enlarge(camera: Camera, depth, camera_ref: Camera, color_ref):
    K, R_c2w, T_c2w, _, _ = fromJSON(camera2view(camera).toJSON(0))
    xyz = reconstrucion(K, R_c2w, T_c2w, depth)
    K_r, R_r, t_r, _, _ = fromJSON(camera2view(camera_ref).toJSON(0))
    uv, z = projection(K_r, R_r, t_r, xyz)
    height, width = color_ref.permute(1, 2, 0).shape[:2]
    uv_idx = uv[..., :2]
    uv_idx = uv_idx.round().type(torch.int64)
    is_edge = uv_idx[..., 1] < 0
    is_edge |= uv_idx[..., 1] >= height - 1
    is_edge |= uv_idx[..., 0] < 0
    is_edge |= uv_idx[..., 0] >= width - 1
    w_enlarge = max(-uv_idx[..., 0].min(), uv_idx[..., 0].max() - width + 1)
    h_enlarge = max(-uv_idx[..., 1].min(), uv_idx[..., 1].max() - height + 1)
    return is_edge, w_enlarge*2/width + 1, h_enlarge*2/height + 1


def show3images(distorted_image, reference_image, warpedref_image):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(ncols=3, nrows=1)
    axs[0].set_title('distorted image')
    axs[0].imshow(distorted_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('reference image')
    axs[1].imshow(reference_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('warped image')
    axs[2].imshow(warpedref_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    fig.tight_layout(pad=5)
    plt.show()


videoout = None


def save2video(distorted_image, warpedref_image):
    import cv2
    frame = torch.concat((distorted_image, warpedref_image), dim=1).permute(1, 2, 0)
    frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    height, width, _ = frame_uint8.shape
    global videoout
    if videoout is None:
        videoout = cv2.VideoWriter('output/run.mp4', -1, 60, (width, height))
        videoout.write((frame.clamp(0, 1) * 255).type(torch.uint8).cpu().numpy())


def save2images(distorted_image, warpedref_image, groundtruth_image, n_frame, n_render, folder="output/run"):
    os.makedirs(folder, exist_ok=True)
    import cv2
    frame = torch.concat((distorted_image, warpedref_image, groundtruth_image), dim=1).permute(1, 2, 0)
    frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(os.path.join(folder, f"frame{n_frame}_{n_render}.png"), frame_uint8)


def save4training(n_frame, n_render, folder="output/run", **kwargs):
    name = f"frame{n_frame}_{n_render}.png"
    import cv2
    for k, img in kwargs.items():
        folder_ = os.path.join(folder, k)
        os.makedirs(folder_, exist_ok=True)
        frame = img.permute(1, 2, 0)
        frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
        cv2.imwrite(os.path.join(folder_, name), frame_uint8)


def mark_visible(camera: Camera, gaussians: GaussianModel, pipeline: PipelineParams):
    view = camera2view(camera)
    # view = camera2view(Camera(
    #     pose=camera.pose,
    #     fovx=camera.fovx*0.5,
    #     fovy=camera.fovy*0.5,
    #     width=camera.width,
    #     height=camera.height
    # ))  # debug
    with torch.no_grad():
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)
        visible = render_pkg["visibility_filter"]
        return visible


def culling(dst_gaussians: GaussianModel, src_gaussians: GaussianModel, visible):
    dst_gaussians._xyz = src_gaussians._xyz[visible, ...]
    dst_gaussians._features_dc = src_gaussians._features_dc[visible, ...]
    dst_gaussians._features_rest = src_gaussians._features_rest[visible, ...]
    dst_gaussians._scaling = src_gaussians._scaling[visible, ...]
    dst_gaussians._rotation = src_gaussians._rotation[visible, ...]
    dst_gaussians._opacity = src_gaussians._opacity[visible, ...]
    dst_gaussians.max_radii2D = src_gaussians.max_radii2D[visible, ...]


def frustum_culling(client_gaussians: GaussianModel, server_gaussians: GaussianModel, camera: Camera, pipeline: PipelineParams):
    visible = mark_visible(camera, server_gaussians, pipeline)
    with torch.no_grad():
        culling(client_gaussians, server_gaussians, visible)


def mark_visible_different(gaussians: VQGaussianModel, last_gaussians: VQGaussianModel, visible):

    def mark_attr_different(threshold, attr: Attribute, i=0):
        attributes = gaussians.get_data(attr, i)
        last_attributes = last_gaussians.get_data(attr, i)
        different = (attributes - last_attributes) > threshold
        if different.dim() > 1:
            different = different.any(dim=-1)
        return torch.logical_and(different,  visible)
    mark = mark_attr_different(0.1, "xyz")
    mark |= mark_attr_different(0.1, Attribute.features_dc)
    mark |= mark_attr_different(0.1, Attribute.features_rest)
    mark |= mark_attr_different(0.1, Attribute.scaling)
    mark |= mark_attr_different(0.1, Attribute.rotation)
    mark |= mark_attr_different(0.1, Attribute.opacity)
    print("visible", visible.sum().item())
    print("visible different", mark.sum().item())
    return mark


def update_visible_different_to_last(gaussians: VQGaussianModel, last_gaussians: VQGaussianModel, visible_different):

    def load_attr_visible_different(attr: Attribute, i=0):
        attributes = gaussians.get_data(attr, i)
        last_attributes = last_gaussians.get_data(attr, i)
        last_attributes[visible_different, ...] = attributes[visible_different, ...]
        last_gaussians.set_data(attr, last_attributes, i)
    load_attr_visible_different("xyz")
    load_attr_visible_different(Attribute.features_dc)
    load_attr_visible_different(Attribute.features_rest)
    load_attr_visible_different(Attribute.scaling)
    load_attr_visible_different(Attribute.rotation)
    load_attr_visible_different(Attribute.opacity)


def load_visible_from_last(gaussians: VQGaussianModel, last_gaussians: VQGaussianModel, visible):

    def load_attr_visible(attr: Attribute, i=0):
        if gaussians.get_data(attr, i).shape[0] == last_gaussians.get_data(attr, i).shape[0]:
            with torch.no_grad():
                setattr(gaussians, "_" + str(attr), getattr(gaussians, "_" + str(attr))[visible, ...].clone())
        gaussians.set_data(attr, last_gaussians.get_data(attr, i)[visible, ...], i)
    load_attr_visible("xyz")
    load_attr_visible(Attribute.features_dc)
    load_attr_visible(Attribute.features_rest)
    load_attr_visible(Attribute.scaling)
    load_attr_visible(Attribute.rotation)
    load_attr_visible(Attribute.opacity)


def compute_next_lod(bitlimit: int, visible: torch.Tensor, should_reload: torch.Tensor, current_lods: torch.Tensor, lod_bitsize: List[int], gaussians: GaussianModel, trace={}):
    # TODO: 计算接下来每个参数都需要多少LoD, 让新增的接近平均LoD
    # 计算: 要重载的gaussians重载到多高lod?
    avg_lod = current_lods[visible].float().mean()  # 所有可见gaussian的平均lod
    def compute_reload_size(n, lod, lod_bitsize): return sum(lod_bitsize[:lod+1])*n  # 用于计算gaussians重载到指定lod需要多少带宽
    reload_lod = 0
    while compute_reload_size(should_reload.sum(), reload_lod+1, lod_bitsize) < bitlimit and reload_lod+1 < min(avg_lod, len(lod_bitsize)):
        reload_lod += 1  # 加重载lod加到满或者超过了平均lod
    print("should reload", should_reload.sum().item(), "avglod", avg_lod.item(), "reload lod", reload_lod)
    trace["should reload"] = should_reload.sum().item()
    trace["avglod"] = avg_lod.item()
    trace["reload lod"] = reload_lod
    reload_need_bit = compute_reload_size(should_reload.sum(), reload_lod, lod_bitsize)  # 完全重载需要多少bit
    if reload_need_bit > bitlimit:  # 重载都不够
        can_reload_n = int(bitlimit / lod_bitsize[0])  # 能重载多少个
        print("cannot full reload", reload_need_bit.item(), ">", bitlimit, "can only reload", can_reload_n, "x", lod_bitsize[0])
        trace["only reload"] = can_reload_n
        should_reload_tmp = should_reload[should_reload]
        score = torch.log(gaussians.get_opacity.detach().squeeze(-1))+torch.sum(gaussians._scaling.detach(), dim=1)
        score = score[should_reload]
        top = torch.topk(score, can_reload_n).values[-1].item()
        should_reload_tmp[score < top] = False  # 删除小透明
        should_reload[should_reload.clone()] = should_reload_tmp
        current_lods[should_reload] = reload_lod  # 重载之
        return current_lods, 0, should_reload  # 重载完就占满了, 结束
    current_lods[should_reload] = reload_lod  # 重载之
    bitlimit_rest = bitlimit - reload_need_bit  # 重载完lod0还剩多大带宽
    print("rest bandwidth", bitlimit_rest.item())
    trace["rest bandwidth"] = bitlimit_rest.item()
    # 计算: 要提升lod的gaussians提升到多高lod?
    while True:
        next_lod = current_lods[visible]+1  # 可见gaussian的下一个lod是?
        can_lift = next_lod < len(lod_bitsize)  # 哪些还能提升(lod最大只有len(lod_bitsize))
        if can_lift.sum() <= 0:  # 没有能提升的了?
            return current_lods, bitlimit_rest, should_reload  # 直接返回
        next_lod_can_lift = next_lod[can_lift]  # 取出还能提升的
        next_lod_can_lift_sorted_idx = next_lod_can_lift.argsort()  # 按LoD排序, 从低lod开始提升
        bit_for_lift = torch.tensor(lod_bitsize)[next_lod_can_lift]  # 提升到下一个lod需要多少bit

        def accu_to_limit(bits, bitlimit: int, i=0, j=None):
            j = j or bits.shape[0]
            if bits[:i].sum() > bitlimit:
                return i
            if bits[:j].sum() < bitlimit:
                return j
            if i == j or i == j-1:
                return i
            k = (i+j)//2
            if bits[:k].sum() > bitlimit:
                return accu_to_limit(bits, bitlimit, i, k)
            else:
                return accu_to_limit(bits, bitlimit, k, j)
        k = accu_to_limit(bit_for_lift[next_lod_can_lift_sorted_idx], bitlimit_rest)  # 找出最多可以提到第几个gaussian
        if k <= 0:
            return current_lods, bitlimit_rest, should_reload
        next_lod_to_lift_sorted_idx = next_lod_can_lift_sorted_idx[:k]  # 待提升的gaussian的index的列表
        bitlimit_rest -= bit_for_lift[next_lod_to_lift_sorted_idx].sum()  # 提升后的剩余带宽
        print("lift", k, "rest bandwidth", bitlimit_rest.item())
        # 操作: 按照上述计算结果填充current_lods
        tmp_visible = current_lods[visible]
        tmp_can_lift = tmp_visible[can_lift]
        tmp_can_lift[next_lod_to_lift_sorted_idx] += 1
        tmp_visible[can_lift] = tmp_can_lift
        current_lods[visible] = tmp_visible
    # TODO: 有空余带宽时预取其他区域


def init_gaussians(gaussians: VQGaussianModel, init_path=None):
    gaussians.load_ply(init_path)

    def init_attr(attr: Attribute, i=0):
        attributes = gaussians.get_data(attr, i)
        attributes[...] = 0
        gaussians.set_data(attr, attributes, i)
    init_attr("xyz")
    init_attr(Attribute.features_dc)
    init_attr(Attribute.features_rest)
    init_attr(Attribute.scaling)
    init_attr(Attribute.rotation)
    init_attr(Attribute.opacity)


lod_log2_clusters = {
    Attribute.features_dc:   [],
    Attribute.features_rest: [],
    Attribute.scaling:       [],
    Attribute.rotation:      [],
    Attribute.opacity:       [],
}

current_log2_clusters = {
    Attribute.features_dc:   4,
    Attribute.features_rest: 0,
    Attribute.scaling:       8,
    Attribute.rotation:      6,
    Attribute.opacity:       4,
}


def lift_attr(attr: Attribute, lod_log2_clusters, current_log2_clusters):
    current_log2_clusters[attr] += 1
    lod_log2_clusters[Attribute.features_dc].append(current_log2_clusters[Attribute.features_dc])
    lod_log2_clusters[Attribute.features_rest].append(current_log2_clusters[Attribute.features_rest])
    lod_log2_clusters[Attribute.scaling].append(current_log2_clusters[Attribute.scaling])
    lod_log2_clusters[Attribute.rotation].append(current_log2_clusters[Attribute.rotation])
    lod_log2_clusters[Attribute.opacity].append(current_log2_clusters[Attribute.opacity])


# 每个LoD都只有一个参数提升
for i in range(2):
    lift_attr(Attribute.scaling, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
for i in range(1, 5):
    lift_attr(Attribute.rotation, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.scaling, lod_log2_clusters, current_log2_clusters)
    if not i % 2 == 0:
        continue
    lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
for i in range(2):
    lift_attr(Attribute.rotation, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.scaling, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
current_log2_clusters[Attribute.features_rest] = 3
lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
for i in range(2):
    lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.rotation, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.rotation, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
for i in range(2):
    lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.features_dc, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
for i in range(2):
    lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.features_rest, lod_log2_clusters, current_log2_clusters)
    lift_attr(Attribute.opacity, lod_log2_clusters, current_log2_clusters)
# 共48个LoD


class LoDLoadConfig(NamedTuple):
    reload_path: str
    n_lod: int
    codebook_dirpath: str


def load_lod(gaussians: VQGaussianModel, current_lods: torch.Tensor, loader: KMeansGaussianModel, config: LoDLoadConfig):
    init_gaussians(gaussians, init_path=config.reload_path)  # 先全清零

    for lod in range(config.n_lod):  # 一个一个LoD的来
        should_load = current_lods == lod  # 有哪些位于这个LoD
        if should_load.sum() <= 0:  # 这个LoD没有参数就退出
            continue

        def load_attr(attr: Attribute, config: LoDLoadConfig, i=0):
            log2_clusters = lod_log2_clusters[attr][lod]  # 这个LoD的这个参数对应哪一级量化
            if log2_clusters <= 0:
                return
            loader.load_codebook(config.codebook_dirpath, log2_clusters=log2_clusters, attr=attr, i=i)  # 加载对应的codebook
            loader.dequantize(attr=attr, quant=loader.quantize(attr=attr, i=i))  # 执行量化
            data = gaussians.get_data(attr=attr, i=i)
            data[should_load] = loader.get_data(attr=attr, i=i)[should_load]  # 给对应的LoD赋值
            gaussians.set_data(attr=attr, kdata=data, i=i)
        loader.load_ply(config.reload_path)
        load_attr(Attribute.features_dc, config)
        load_attr(Attribute.features_rest, config)
        load_attr(Attribute.scaling, config)
        load_attr(Attribute.rotation, config)
        load_attr(Attribute.opacity, config)

    culling(gaussians, gaussians, current_lods >= 0)  # 裁剪掉没有LoD的
    gaussians._xyz = loader._xyz[current_lods >= 0]
    return gaussians


def rotate_speed(Rs: torch.Tensor):
    Qs = matrix_to_quaternion(Rs)
    speeds = Qs.max(dim=0).values - Qs.min(dim=0).values
    return speeds


n_lod = 32
lod_bitsize = [sum(attr[0] for attr in lod_log2_clusters.values())]
lod_bitsize += [sum(attr[i] - attr[i-1] for attr in lod_log2_clusters.values()) for i in range(1, n_lod)]
bitlimit = 5*2**20  # 5x30Mbps & 30FPS

if __name__ == "__main__":
    torch.device("cuda").__enter__()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    bandwidth = pd.read_csv(args.bandwidth)["throughput_mbps"][args.bandwidth_start:(args.bandwidth_end or args.bandwidth_start+args.max_frame)]
    bandwidth_iter = cycle(bandwidth)
    prediction = prediction_dict[args.prediction](**json.loads(args.prediction_conf))
    if args.prediction_load:
        prediction.load(args.prediction_load)
    prediction_fov = LinearPredictionFoV(path=args.fov_save)
    pose_dataset = CameraPoseDataset(args.cameras, history_size=args.history_size, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
    frame_stride = 1/args.fps
    last_frame = None
    server_gaussians = GaussianModel(args.sh_degree)
    client_gaussians = KMeansGaussianModel(args.sh_degree)
    client_gaussians_vqloader = KMeansGaussianModel(args.sh_degree)
    last_gaussians = KMeansGaussianModel(args.sh_degree)
    restoration = get_restoration(args.restore_save, args)
    current_lods = None
    traces = []
    for i in range(len(pose_dataset) - args.cameras_start):
        (pose_history, pose_groundtruth) = pose_dataset[i+args.cameras_start]
        n_frame = i + 1
        timestamp = pose_history["timestamp"][0].item()
        trace = {"timestamp": timestamp}
        if n_frame > args.max_frame:
            break
        frame_folder = os.path.join(args.video, f"frame{n_frame}", "point_cloud")
        n_iter = searchForMaxIteration(frame_folder)
        frame_ply = os.path.join(frame_folder, f"iteration_{n_iter}", "point_cloud.ply")
        print(f"{timestamp:.4f}", "frame", n_frame, "loading")

        pose_prediction = predict_camera(
            prediction, pose_history,
            prediction_stride=args.prediction_stride, prediction_length=args.prediction_length,
            fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
        print("loss", torch.abs(pose_prediction["R"] - pose_groundtruth["R"]).mean(), torch.abs(pose_prediction["T"] - pose_groundtruth["T"]).mean())
        trace["prediction loss"] = {
            "R": torch.abs(pose_prediction["R"] - pose_groundtruth["R"]).mean().item(),
            "T": torch.abs(pose_prediction["T"] - pose_groundtruth["T"]).mean().item(),
        }
        prediction_camera = Camera(
            pose=Pose(
                timestamp=pose_history["timestamp"][-1],
                R=pose_prediction["R"][-1, ...],
                T=pose_prediction["T"][-1, ...]),
            fovx=args.fovx, fovy=args.fovy, width=args.width//4, height=args.height//4)
        speed = rotate_speed(pose_prediction["R"])
        w_enlarge_pred, h_enlarge_pred = prediction_fov.predict(speed)
        prediction_camera_enlarged = Camera(
            pose=Pose(
                timestamp=pose_history["timestamp"][-1],
                R=pose_prediction["R"][-1, ...],
                T=pose_prediction["T"][-1, ...]),
            fovx=math.atan(w_enlarge_pred*math.tan(args.fovx/2))*2, fovy=math.atan(h_enlarge_pred*math.tan(args.fovy/2))*2,
            width=args.width//4, height=args.height//4)
        trace["enlarge prediction"] = dict(
            fovx=math.atan(w_enlarge_pred*math.tan(args.fovx/2))*2,
            fovy=math.atan(h_enlarge_pred*math.tan(args.fovy/2))*2,
            w=w_enlarge_pred.item(),
            h=w_enlarge_pred.item()
        )

        # 服务端渲染
        server_gaussians.load_ply(path=frame_ply)
        reference_image, _ = render_frame(prediction_camera, server_gaussians, pipeline)
        reference_image_enlarged, _ = render_frame(prediction_camera_enlarged, server_gaussians, pipeline)

        # 发送计算初始化
        if n_frame == 1:
            init_gaussians(last_gaussians, frame_ply)
            current_lods = torch.zeros(last_gaussians._xyz.shape[0], dtype=torch.int, device=last_gaussians._xyz.device)-1

        # 发送计算
        client_gaussians.load_ply(path=frame_ply)
        visible = mark_visible(prediction_camera, client_gaussians, pipeline)
        if args.use_enlarged_in_mark_visible:
            visible = mark_visible(prediction_camera_enlarged, client_gaussians, pipeline)
        should_reload = mark_visible_different(client_gaussians, last_gaussians, visible)
        current_bitlimit = bandwidth_iter.__next__() * 2**20 / args.fps  # 读取带宽数据
        trace["bitlimit"] = current_bitlimit
        current_lods, bitlimit_rest, should_reload = compute_next_lod(current_bitlimit, visible, should_reload, current_lods, lod_bitsize, client_gaussians, trace)  # 决策量化级别, 预测视角内塞满带宽
        update_visible_different_to_last(client_gaussians, last_gaussians, should_reload)
        # load_visible_from_last(client_gaussians, last_gaussians, visible)  # debug
        load_lod(client_gaussians, current_lods, client_gaussians_vqloader,
                 config=LoDLoadConfig(reload_path=frame_ply, n_lod=n_lod, codebook_dirpath=args.codebooks))  # 按照量化级别执行反量化
        total_missing_pixels = 0
        w_enlarge, h_enlarge = 0, 0
        total_missing_pixels_enlarged = 0
        trace["rendering"] = []
        for j in range(args.prediction_length):
            render_trace = {}
            n_render = j + 1
            print(f"{pose_groundtruth['timestamp'][j].item():.4f}", "frame", n_frame, "rendering", n_render)
            render_trace["timestamp"] = pose_groundtruth['timestamp'][j].item()

            groundtruth_camera = Camera(
                pose=Pose(
                    timestamp=pose_groundtruth["timestamp"][j].item(),
                    R=pose_groundtruth["R"][j, ...],
                    T=pose_groundtruth["T"][j, ...]),
                fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
            distorted_image, depth = render_frame(groundtruth_camera, client_gaussians, pipeline)
            warpedref_image, is_edge = warping_frame(groundtruth_camera, depth[0, ...], prediction_camera, reference_image)
            warpedenlargedref_image, is_edge_enlarged = warping_frame(groundtruth_camera, depth[0, ...], prediction_camera_enlarged, reference_image_enlarged)
            is_edge, w_enlarge_, h_enlarge_ = compute_enlarge(groundtruth_camera, depth[0, ...], prediction_camera, reference_image)
            w_enlarge = max(w_enlarge, w_enlarge_)
            h_enlarge = max(h_enlarge, h_enlarge_)
            print("Missing pixels", is_edge.sum().item())
            render_trace["missing pixels"] = is_edge.sum().item()
            total_missing_pixels += is_edge.sum().item()
            print("Missing pixels after predicted enlarge", is_edge_enlarged.sum().item())
            total_missing_pixels_enlarged += is_edge_enlarged.sum().item()
            render_trace["missing pixels after enlarge"] = is_edge_enlarged.sum().item()

            enlarge_camera = Camera(
                pose=Pose(
                    timestamp=pose_history["timestamp"][-1],
                    R=pose_prediction["R"][-1, ...],
                    T=pose_prediction["T"][-1, ...]),
                fovx=math.atan(w_enlarge_*math.tan(args.fovx/2))*2, fovy=math.atan(h_enlarge_*math.tan(args.fovy/2))*2,
                width=args.width//4, height=args.height//4)
            render_trace["enlarge groundtruth"] = dict(
                fovx=math.atan(w_enlarge_*math.tan(args.fovx/2))*2,
                fovy=math.atan(h_enlarge_*math.tan(args.fovy/2))*2,
                w=w_enlarge_.item(),
                h=h_enlarge_.item()
            )
            enlargeref_image, _ = render_frame(enlarge_camera, server_gaussians, pipeline)
            warpedenlargtref_image, is_edge = warping_frame(groundtruth_camera, depth[0, ...], enlarge_camera, enlargeref_image)
            print("Missing pixels after groundtruth enlarge", is_edge.sum().item())

            groundtruth_image, _ = render_frame(groundtruth_camera, server_gaussians, pipeline)
            # show3images(distorted_image, reference_image, warpedref_image)  # debug
            # save2video(distorted_image, warpedref_image)  # debug
            # save2images(distorted_image, warpedref_image, warpedenlargtref_image, n_frame, n_render)  # debug
            restored_image = restoration.restore(distorted_image, warpedref_image)  # 色彩恢复, 用训好的merge模型直接出
            enlargerestored_image = restoration.restore(distorted_image, warpedenlargedref_image)  # 色彩恢复, 用训好的merge模型直接出
            render_trace["restored_quality"] = metrics(restored_image, groundtruth_image)  # 测质量
            render_trace["enlargerestored_quality"] = metrics(enlargerestored_image, groundtruth_image)  # 测质量
            save4training(
                n_frame, n_render, folder=args.image_save,
                distorted=distorted_image,
                warped=warpedref_image,
                warpedenlarged=warpedenlargedref_image,
                groundtruth=groundtruth_image,
                restored=restored_image,
                enlargerestored=enlargerestored_image,
            )
            trace["rendering"].append(render_trace)
        print("fovx", args.fovx, "->", math.atan(w_enlarge*math.tan(args.fovx)))
        print("fovy", args.fovy, "->", math.atan(h_enlarge*math.tan(args.fovy)))
        trace["enlarge groundtruth"] = dict(
            fovx=math.atan(w_enlarge*math.tan(args.fovx/2))*2,
            fovy=math.atan(h_enlarge*math.tan(args.fovy/2))*2,
            w=w_enlarge.item(),
            h=h_enlarge.item()
        )
        print("speed", speed, "enlarge", w_enlarge.item(), h_enlarge.item(), "Missing", total_missing_pixels)
        trace["missing pixels"] = total_missing_pixels
        print("predicted enlarge", w_enlarge_pred.item(), h_enlarge_pred.item(), "Missing", total_missing_pixels_enlarged)
        trace["missing pixels after enlarge"] = total_missing_pixels_enlarged
        with open(args.fov_save, "a", encoding="utf8") as f:
            f.write(f"{w_enlarge}, {h_enlarge}, " + ', '.join([str(i) for i in speed.cpu().numpy().tolist()]) + '\n')
        traces.append(trace)
        with open(args.trace_save, "w", encoding='utf8') as f:
            json.dump(traces, f, indent=2)
    if videoout is not None:
        videoout.release()
