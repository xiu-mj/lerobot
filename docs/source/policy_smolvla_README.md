## Paper

https://arxiv.org/abs/2506.01844

## Rectified Flow Variant

This fork includes a configurable Rectified Flow objective for SmolVLA. See
[`smolvla_rectified_flow_runbook.md`](./smolvla_rectified_flow_runbook.md) for training,
evaluation, and server-sync commands. See
[`smolvla_rectified_flow_code_changes.md`](./smolvla_rectified_flow_code_changes.md) for
the implementation summary.

## Adaptive Chunk and Adaptive Step

The Rectified Flow variant can optionally predict both the generated action horizon and the
number of flow inference steps from a shared task-context representation. See
[`smolvla_adaptive_computation.md`](./smolvla_adaptive_computation.md) for configuration,
offline label construction, controller training, and fixed-budget ablations.

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```
