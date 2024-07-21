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
from warping import fromJSON, reconstrucion, projection, warp

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=1.4773902348773813, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=1.2005465997792715, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--buffersize", type=int, default=10, help="Camera buffer size.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)
parser.add_argument("--prediction-stride", type=int, default=5)
parser.add_argument("--prediction-length", type=int, default=5)
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--max-frame", type=int, default=30)


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


def predict_camera(pose_history, fovx: float, fovy: float, width: int, height: int):
    return Camera(
        pose=Pose(
            timestamp=pose_history["timestamp"][-1],
            R=pose_history["R"][-1, ...],
            T=pose_history["T"][-1, ...]),
        fovx=fovx + 0.1, fovy=fovy + 0.1, width=width//4, height=height//4)


def render_frame(camera: Camera, gaussians: GaussianModel, pipeline: PipelineParams):
    view = camera2view(camera)
    with torch.no_grad():
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        return rendering, depth


def warping_frame(camera: Camera, depth, camera_ref: Camera, color_ref):
    K, R_c2w, T_c2w, _, _ = fromJSON(camera2view(camera).toJSON(0))
    xyz = reconstrucion(K, R_c2w, T_c2w, depth)
    K_r, R_r, t_r, _, _ = fromJSON(camera2view(camera_ref).toJSON(0))
    uv, z = projection(K_r, R_r, t_r, xyz)
    warped = warp(uv, color_ref, z)  # wrap it
    return warped


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


def save2images(distorted_image, warpedref_image, n_frame, n_render, folder="output/run"):
    os.makedirs(folder, exist_ok=True)
    import cv2
    frame = torch.concat((distorted_image, warpedref_image), dim=1).permute(1, 2, 0)
    frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(os.path.join(folder, f"frame{n_frame}_{n_render}.png"), frame_uint8)


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


def culling(client_gaussians: GaussianModel, server_gaussians: GaussianModel, visible):
    client_gaussians._xyz = server_gaussians._xyz[visible, ...]
    client_gaussians._features_dc = server_gaussians._features_dc[visible, ...]
    client_gaussians._features_rest = server_gaussians._features_rest[visible, ...]
    client_gaussians._scaling = server_gaussians._scaling[visible, ...]
    client_gaussians._rotation = server_gaussians._rotation[visible, ...]
    client_gaussians._opacity = server_gaussians._opacity[visible, ...]
    client_gaussians.max_radii2D = server_gaussians.max_radii2D[visible, ...]


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


def compute_next_lod(bitlimit: int, visible: torch.Tensor, visible_different: torch.Tensor, current_lods: torch.Tensor, lod_bitsize: List[int]):
    # TODO: 计算接下来每个参数都需要多少LoD, 让新增的接近平均LoD
    # 计算: 要重载的gaussians重载到多高lod?
    should_reload = visible_different  # 视野中的变化区域需要重载
    avg_lod = current_lods[visible].float().mean()  # 所有可见gaussian的平均lod
    def compute_reload_size(n, lod, lod_bitsize): return sum(lod_bitsize[:lod+1])*n  # 用于计算gaussians重载到指定lod需要多少带宽
    reload_lod = 0
    while compute_reload_size(should_reload.sum(), reload_lod, lod_bitsize) < bitlimit and reload_lod <= avg_lod:
        reload_lod += 1  # 加重载lod加到满或者超过了平均lod
    current_lods[should_reload] = reload_lod  # 重载之
    print("should reload", should_reload.sum().item(), "avglod", avg_lod.item(), "reload lod", reload_lod)
    bitlimit_rest = bitlimit - compute_reload_size(should_reload.sum(), reload_lod, lod_bitsize)  # 重载完lod0还剩多大带宽
    print("rest bandwidth", bitlimit_rest.item())
    # TODO: 处理负值(lod0都传不完)
    # 计算: 要提升lod的gaussians提升到多高lod?
    next_lod = current_lods[visible]+1  # 可见gaussian的下一个lod是?
    can_lift = next_lod < len(lod_bitsize)  # 哪些还能提升(lod最大只有len(lod_bitsize))
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
    next_lod_to_lift_sorted_idx = next_lod_can_lift_sorted_idx[:k]  # 待提升的gaussian的index的列表
    bitlimit_rest -= bit_for_lift[next_lod_to_lift_sorted_idx].sum()  # 提升后的剩余带宽
    print("lift", k, "rest bandwidth", bitlimit_rest.item())
    # TODO: 有空余带宽时预取其他区域
    # 操作: 按照上述计算结果填充current_lods
    tmp_visible = current_lods[visible]
    tmp_can_lift = tmp_visible[can_lift]
    tmp_can_lift[next_lod_to_lift_sorted_idx] += 1
    tmp_visible[can_lift] = tmp_can_lift
    current_lods[visible] = tmp_visible
    return current_lods, bitlimit_rest


def load_lod(gaussians: VQGaussianModel, current_lods):
    # TODO: load VQ LoD 数据
    pass


def compute_visible_different(gaussians: VQGaussianModel, last_gaussians: VQGaussianModel, camera: Camera, pipeline: PipelineParams):
    visible = mark_visible(camera, gaussians, pipeline)
    visible_different = mark_visible_different(gaussians, last_gaussians, visible)
    update_visible_different_to_last(gaussians, last_gaussians, visible_different)
    load_visible_from_last(gaussians, last_gaussians, visible)
    return visible, visible_different


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


lod_bitsize = [8, 1, 1, 1, 1, 1, 1, 1, 1]
bitlimit = 2e6

if __name__ == "__main__":
    torch.device("cuda").__enter__()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    pose_dataset = CameraPoseDataset(args.cameras, history_size=args.history_size, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
    frame_stride = 1/args.fps
    last_frame = None
    server_gaussians = GaussianModel(args.sh_degree)
    client_gaussians = KMeansGaussianModel(args.sh_degree)
    last_gaussians = KMeansGaussianModel(args.sh_degree)
    current_lods = None
    for i, (pose_history, pose_groundtruth) in enumerate(pose_dataset):
        n_frame = i + 1
        timestamp = pose_history["timestamp"][0].item()
        if n_frame > args.max_frame:
            break
        print(f"{timestamp:.4f}", "frame", n_frame, "loading")
        prediction_camera = predict_camera(pose_history, fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
        frame_folder = os.path.join(args.video, f"frame{n_frame}", "point_cloud")
        n_iter = searchForMaxIteration(frame_folder)
        frame_ply = os.path.join(frame_folder, f"iteration_{n_iter}", "point_cloud.ply")

        # 服务端渲染
        server_gaussians.load_ply(path=frame_ply)
        reference_image, _ = render_frame(prediction_camera, server_gaussians, pipeline)

        # 发送计算初始化
        if n_frame == 1:
            init_gaussians(last_gaussians, frame_ply)
            current_lods = torch.zeros(last_gaussians._xyz.shape[0], dtype=torch.int, device=last_gaussians._xyz.device)-1

        # 发送计算
        client_gaussians.load_ply(path=frame_ply)
        visible, visible_different = compute_visible_different(client_gaussians, last_gaussians, prediction_camera, pipeline)
        # TODO: 读取带宽数据
        current_lods, bitlimit_rest = compute_next_lod(bitlimit, visible, visible_different, current_lods, lod_bitsize)  # 决策量化级别, 预测视角内塞满带宽
        # TODO: 按照量化级别执行反量化
        for j in range(args.prediction_length):
            n_render = j + 1
            print(f"{pose_groundtruth['timestamp'][j].item():.4f}", "frame", n_frame, "rendering", n_render)
            groundtruth_camera = Camera(
                pose=Pose(
                    timestamp=pose_groundtruth["timestamp"][j].item(),
                    R=pose_groundtruth["R"][j, ...],
                    T=pose_groundtruth["T"][j, ...]),
                fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
            distorted_image, depth = render_frame(groundtruth_camera, client_gaussians, pipeline)
            warpedref_image = warping_frame(groundtruth_camera, depth[0, ...], prediction_camera, reference_image.permute(1, 2, 0)).permute(2, 0, 1)
            # show3images(distorted_image, reference_image, warpedref_image)  # debug
            # save2video(distorted_image, warpedref_image)  # debug
            save2images(distorted_image, warpedref_image, n_frame, n_render)  # debug
            # TODO: 色彩恢复
            # TODO: 测质量

    if videoout is not None:
        videoout.release()
