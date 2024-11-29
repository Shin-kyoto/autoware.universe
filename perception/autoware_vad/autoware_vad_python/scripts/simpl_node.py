import os.path as osp

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import TrackedObjects
from autoware_simpl_python.checkpoint import load_checkpoint
from autoware_simpl_python.conversion import convert_lanelet
from autoware_simpl_python.conversion import convert_transform_stamped
from autoware_simpl_python.conversion import from_tracked_objects
from autoware_simpl_python.conversion import sort_object_infos
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
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import torch
import yaml

from .node_utils import ModelInput
from .node_utils import softmax


class SimplNode(Node):
    def __init__(self) -> None:
        super().__init__("simpl_python_node")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # subscribers
        self._object_sub = self.create_subscription(
            TrackedObjects,
            "~/input/objects",
            self._callback,
            qos_profile,
        )

        # publisher
        self._object_pub = self.create_publisher(PredictedObjects, "~/output/objects", qos_profile)

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

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _callback(self, msg: TrackedObjects) -> None:
        current_ego = self._get_current_ego(msg.header)

        if current_ego is None:
            self.get_logger().warn("No ego found.")
            return

        # remove ancient agent history
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        states, infos = from_tracked_objects(msg)
        if len(states) == 0:  # TODO: publish empty msg
            self.get_logger().warn("No agent found.")
            return

        self._history.update(states, infos)

        # inference
        inputs = self._preprocess(self._history, current_ego, self._lane_segments)
        # self.get_logger().info(f"{inputs.actor[0]=}")
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
        pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)

        # TODO(ktro2828): guarantee the order of agent info and history is same.
        infos = sort_object_infos(self._history.infos, inputs.uuids)

        # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=infos,
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            score_threshold=self._score_threshold,
        )
        self._object_pub.publish(pred_objs)

    def _get_current_ego(self, header: Header) -> AgentState | None:
        """Return the current ego state. If failed to listen to transform, returns None.

        Args:
            header (Header): Current message header.

        Returns:
            AgentState | None: Returns AgentState if succeeded to listen transform.
        """
        to_frame = "map"
        from_frame = "base_link"

        # TODO(ktro2828): update values
        uuid = "Ego"
        label_id = ObjectClassification.CAR
        size = np.array((0, 0, 0))
        vxy = np.array((0.0, 0.0))

        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                to_frame, from_frame, header.stamp, rclpy.duration.Duration(seconds=0.1)
            )
            return convert_transform_stamped(tf_stamped, uuid, label_id, size, vxy)
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform {to_frame} to {from_frame}: {ex}")
            return None

    def _preprocess(
        self,
        history: AgentHistory,
        current_ego: AgentState,
        lane_segments: list[LaneSegment],
    ) -> ModelInput:
        """Run preprocess.

        Args:
            history (AgentHistory): Agent history.
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

    node = SimplNode()
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
