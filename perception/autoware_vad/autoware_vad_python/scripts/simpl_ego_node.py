import hashlib
import os.path as osp

from autoware_perception_msgs.msg import PredictedObjects
from autoware_simpl_python.checkpoint import load_checkpoint
from autoware_simpl_python.conversion import convert_lanelet
from autoware_simpl_python.conversion import from_odometry
from autoware_simpl_python.conversion import timestamp2ms
from autoware_simpl_python.conversion import to_predicted_objects
from autoware_simpl_python.dataclass import AgentHistory
from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import LaneSegment
from autoware_simpl_python.datatype import AgentLabel
from autoware_simpl_python.geometry import rotate_along_z
from autoware_simpl_python.model import Simpl
from autoware_simpl_python.preprocess import embed_agent
from autoware_simpl_python.preprocess import embed_polyline
from autoware_simpl_python.preprocess import relative_pose_encode
from nav_msgs.msg import Odometry
import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession
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
        self._lane_segments: list[LaneSegment] = convert_lanelet(lanelet_file)
        self._history = AgentHistory(max_length=num_timestamp)

        self._ego_uuid = hashlib.shake_256("EGO".encode()).hexdigest(8)

        self._label_ids = [AgentLabel.from_str(label).value for label in labels]

        # onnx inference
        self._is_onnx = osp.splitext(model_path)[-1] == ".onnx"
        if self._is_onnx:
            self._session = InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
        else:
            model_config_path = (
                self.declare_parameter("model_config", descriptor=descriptor)
                .get_parameter_value()
                .string_value
            )
            with open(model_config_path) as f:
                model_config = yaml.safe_load(f)
            model = Simpl(**model_config)
            model = load_checkpoint(model, model_path)
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

        # pre-process
        inputs = self._preprocess(self._history, current_ego, self._lane_segments)

        # inference
        if self._is_onnx:
            inputs = {name: getattr(inputs, name) for name in self._input_names}
            pred_scores, pred_trajs = self._session.run(self._output_names, inputs)
        else:
            inputs = inputs.cuda()
            with torch.no_grad():
                pred_scores, pred_trajs = self._model(
                    inputs.actor,
                    inputs.lane,
                    inputs.rpe,
                    inputs.rpe_mask,
                )
        # post-process
        pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)

        # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=[info],
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            score_threshold=self._score_threshold,
        )
        self._publisher.publish(pred_objs)

    def _preprocess(
        self,
        history: AgentHistory,
        current_ego: AgentState,
        lane_segments: list[LaneSegment],
    ) -> ModelInput:
        """Run preprocess.

        Args:
            history (AgentHistory): Ego history.
            current_ego (AgentState): Current ego state.
            lane_segments (list[LaneSegments]): Lane segments.

        Returns:
            ModelInput: Model inputs.
        """
        trajectory, uuids = history.as_trajectory()
        agent, agent_ctr, agent_vec = embed_agent(trajectory, current_ego, self._label_ids)
        lane, lane_ctr, lane_vec = embed_polyline(lane_segments, current_ego)

        rpe, rpe_mask = relative_pose_encode(agent_ctr, agent_vec, lane_ctr, lane_vec)

        return ModelInput(uuids, agent, lane, rpe, rpe_mask)

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
