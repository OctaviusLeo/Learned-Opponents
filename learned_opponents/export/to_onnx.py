"""ONNX export.

Responsibility: single command that exports a checkpoint's Policy to an
ONNX_Model whose input is the ``N_rays + 5`` Observation and whose output is
the 2-dim Action (conforming to the Obs_Action_Spec); save a fixed set of
Observations and the Python Policy's Actions for them to ``results/parity/``.

Requirements: 7.1, 7.2, 7.6.

TODO: implement in tasks.md task 13.1 (deferred). Placeholder for now.
"""

raise NotImplementedError("TODO: implement ONNX export — see tasks.md task 13.1")
