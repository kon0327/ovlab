# OpenVLA integration support

Reusable Python implementation shared by the OpenVLA, OpenVLA-OFT and
OpenVLA-QuIC plugin projects. This is a library, not a registered model plugin.
It owns common action/observation helpers and shared training/checkpoint code;
it does not register policies or select experiments.

Each plugin explicitly declares this project as the `openvla-support` source
and lists it in `control_sources` and its build recipes. Host orchestration
loads that source without installing model dependencies. Frozen configuration
bundles include the selected control source. Images install this distribution
before the model-specific plugin package.

The checked-in plugins use a local source for development. When publishing a
plugin independently, replace that source with a pinned Git URL and full commit
(or publish this Python distribution and adapt the build accordingly). Remote
plugins cannot depend on an undeclared local sibling directory.
