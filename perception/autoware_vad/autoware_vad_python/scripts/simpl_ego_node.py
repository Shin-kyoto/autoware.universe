import hashlib
import os.path as osp

from autoware_perception_msgs.msg import PredictedObjects
from autoware_simpl_python.checkpoint import load_checkpoint
from autoware_simpl_python.conversion import from_odometry
from autoware_simpl_python.conversion import timestamp2ms
from autoware_simpl_python.conversion import to_predicted_objects
from autoware_simpl_python.dataclass import AgentHistory
from autoware_simpl_python.datatype import AgentLabel
from autoware_simpl_python.geometry import rotate_along_z
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import numpy as np
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
import rclpy.duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import rclpy.parameter
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import torch
import yaml

from mmcv import Config
from mmcv.runner import wrap_fp16_model
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

from .node_utils import ModelInput
from .node_utils import softmax


class SimplEgoNode(Node):
    """A ROS 2 node to predict EGO trajectory."""

    def __init__(self) -> None:
        super().__init__("simpl_python_ego_node")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # subscribers
        self._subscription = self.create_subscription(
            Odometry,
            "~/input/ego",
            self._callback,
            qos_profile,
        )

        self._subscription = self.create_subscription(
            Image,
            "~/input/image",
            self._callback,
            qos_profile,
        )

        # publisher
        self._publisher = self.create_publisher(PredictedObjects, "~/output/objects", qos_profile)

        # ROS parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)
        num_timestamp = (
            self.declare_parameter("num_timestamp", descriptor=descriptor)
            .get_parameter_value()
            .integer_value
        )
        self._timestamp_threshold = (
            self.declare_parameter("timestamp_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )
        self._score_threshold = (
            self.declare_parameter("score_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )
        lanelet_file = (
            self.declare_parameter("lanelet_file", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )
        labels = (
            self.declare_parameter("labels", descriptor=descriptor)
            .get_parameter_value()
            .string_array_value
        )
        model_path = (
            self.declare_parameter("model_path", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )
        build_only = (
            self.declare_parameter("build_only", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        # input attributes
        self._history = AgentHistory(max_length=num_timestamp)

        self._ego_uuid = hashlib.shake_256("EGO".encode()).hexdigest(8)

        self._label_ids = [AgentLabel.from_str(label).value for label in labels]

        # onnx inference
        self._is_onnx = osp.splitext(model_path)[-1] == ".onnx"
        if True:
            model_config_path = (
                self.declare_parameter("model_config", descriptor=descriptor)
                .get_parameter_value()
                .string_value
            )
            with open(model_config_path) as f:
                model_config = yaml.safe_load(f)

            config_path = "../autoware_vad_python/projects/configs/VAD/VAD_base_e2e.py"
            cfg = Config.fromfile(config_path)
            # import modules from string list.
            if cfg.get('custom_imports', None):
                from mmcv.utils import import_modules_from_strings
                import_modules_from_strings(**cfg['custom_imports'])

            # import modules from plguin/xx, registry will be updated
            if hasattr(cfg, 'plugin'):
                if cfg.plugin:
                    import importlib
                    if hasattr(cfg, 'plugin_dir'):
                        plugin_dir = cfg.plugin_dir
                        _module_dir = osp.dirname(plugin_dir)
                        _module_dir = _module_dir.split('/')
                        _module_path = _module_dir[0]

                        for m in _module_dir[1:]:
                            _module_path = _module_path + '.' + m
                        print(_module_path)
                        plg_lib = importlib.import_module(_module_path)
                    else:
                        # import dir is the dirpath for the config file
                        _module_dir = osp.dirname(config_path)
                        _module_dir = _module_dir.split('/')
                        _module_path = _module_dir[0]
                        for m in _module_dir[1:]:
                            _module_path = _module_path + '.' + m
                        print(_module_path)
                        plg_lib = importlib.import_module(_module_path)

            # set cudnn_benchmark
            if cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True

            cfg.model.pretrained = None
            # in case the test dataset is concatenated
            samples_per_gpu = 1
            if isinstance(cfg.data.test, dict):
                cfg.data.test.test_mode = True
                samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
                if samples_per_gpu > 1:
                    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                    cfg.data.test.pipeline = replace_ImageToTensor(
                        cfg.data.test.pipeline)
            elif isinstance(cfg.data.test, list):
                for ds_cfg in cfg.data.test:
                    ds_cfg.test_mode = True
                samples_per_gpu = max(
                    [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
                if samples_per_gpu > 1:
                    for ds_cfg in cfg.data.test:
                        ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
                        
            # build the model and load checkpoint
            cfg.model.train_cfg = None
            model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            checkpoint_path = "../autoware_vad_python/ckpt/resnet50-19c8e357.pth"
            checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

            # old versions did not save class info in checkpoints, this walkaround is
            # for backward compatibility
            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES
            # palette for visualization in segmentation tasks
            if 'PALETTE' in checkpoint.get('meta', {}):
                model.PALETTE = checkpoint['meta']['PALETTE']
            elif hasattr(dataset, 'PALETTE'):
                # segmentation dataset has `PALETTE` attribute
                model.PALETTE = dataset.PALETTE

            self._model = model.cuda().eval()

        if build_only:
            self.get_logger().info("Model has been built successfully and exit.")
            exit(0)

    def _callback(self, msg: Odometry) -> None:
        # remove invalid ancient agent history
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        current_ego, info = from_odometry(
            msg,
            uuid=self._ego_uuid,
            label_id=AgentLabel.VEHICLE,
            size=(4.0, 2.0, 1.0),  # size is unused dummy
        )
        self._history.update_state(current_ego, info)



        inputs = inputs.cuda()
        with torch.no_grad():
            pred_scores, pred_trajs = self._model(
                return_loss=True,
                img_metas=img_metas,
                ego_his_trajs=ego_his_trajs,
            )
        # post-process
        bbox_results = self._postprocess(pred_scores, pred_trajs)
        pred_trajs = bbox_results['ego_fut_preds']
        # bbox_result['ego_fut_cmd']
        pred_scores = # TODO

        # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=[info],
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            score_threshold=self._score_threshold,
        )
        self._publisher.publish(pred_objs)


    def _postprocess(
        self,
        pred_scores: NDArray | torch.Tensor,
        pred_trajs: NDArray | torch.Tensor,
    ) -> tuple[NDArray, NDArray]:
        """Run postprocess.

        Args:
            pred_scores (NDArray | torch.Tensor): Predicted scores in the shape of
                (N, M).
            pred_trajs (NDArray | torch.Tensor): Predicted trajectories in the shape of
                (N, M, T, 4).

        Returns:
            tuple[NDArray, NDArray]: Transformed and sorted prediction.
        """
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().detach().numpy()
        if isinstance(pred_trajs, torch.Tensor):
            pred_trajs = pred_trajs.cpu().detach().numpy()

        num_agent, num_mode, num_future, num_feat = pred_trajs.shape
        assert num_feat == 4, f"Expected predicted feature is (x, y, vx, vy), but got {num_feat}"

        # transform from agent centric coords to world coords
        current_agent, _ = self._history.as_trajectory(latest=True)
        pred_trajs[..., :2] = rotate_along_z(
            pred_trajs.reshape(num_agent, -1, num_feat)[..., :2], -current_agent.yaw
        ).reshape(num_agent, num_mode, num_future, 2)
        pred_trajs[..., :2] += current_agent.xy[:, None, None, :]

        # sort by score
        pred_scores = softmax(pred_scores, axis=1)
        sort_indices = np.argsort(-pred_scores, axis=1)
        pred_scores = np.take_along_axis(pred_scores, sort_indices, axis=1)
        pred_trajs = np.take_along_axis(pred_trajs, sort_indices[..., None, None], axis=1)

        return pred_scores, pred_trajs


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SimplEgoNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
