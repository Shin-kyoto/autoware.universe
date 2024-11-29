# autoware_vad_python

## Dependencies

```shell
python3 -m pip install onnxruntime-gpu torch
```

## Input / Output

### Input

| Name            | Type                                            | Description           |
| --------------- | ----------------------------------------------- | --------------------- |
| `input/objects` | `autoware_perception_msgs::msg::TrackedObjects` | Tracked objects       |
| `input/ego`     | `nav2_msgs::msg::Odometry`                      | Ego vehicle odomoetry |

### Output

| Name             | Type                                              | Description       |
| ---------------- | ------------------------------------------------- | ----------------- |
| `output/objects` | `autoware_perception_msgs::msg::PredictedObjects` | Predicted objects |

## Parameters

| Name                  | Type   | Description                                                                            | Default value                 |
| --------------------- | ------ | -------------------------------------------------------------------------------------- | ----------------------------- |
| `model_path`          | string | Model onnx file path                                                                   | `$(var data_path)/simpl.onnx` |
| `lanelet_file`        | string | Path to lanelet osm file                                                               | `lanelet2_map.osm`            |
| `num_timestamp`       | int    | The max number of history length                                                       | 48                            |
| `timestamp_threshold` | float  | The theshold value to filter out ancient agent history in [ms]                         | 150                           |
| `build_only`          | bool   | Whether to build only model. If true, node will be terminated after the model is built | false                         |

## Citation

```latex
@article{zhang2024simpl,
      title={SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline for Autonomous Driving},
      author={Lu Zhang and Peiliang Li and Sikang Liu and Shaojie Shen},
      year={2024}
}
```
