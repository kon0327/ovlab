# OVLAB Remote Policy

`ovlab-remote-policy` provides the generic, versioned local RPC boundary used
when an OVLAB runner and a policy require incompatible Python environments.
Evaluation uses an `AF_UNIX` socket with length-prefixed JSON frames. Persistent
model deployments expose the same protocol through `TcpPolicyClient(host, port)`.
The TCP endpoint is OVLAB RPC rather than HTTP. A persistent service keeps its
model loaded across client sessions, releases an unfinished episode when its
client disconnects, and accepts health probes while an inference session is active.
Only one client owns an inference session at a time. Policy
services remain ordinary `PolicyAdapter` implementations behind the boundary;
the runner uses `RemotePolicyAdapter` and contains no model-specific logic.

The prediction schema deliberately carries only identifiers, the authoritative
instruction, and `camera.primary.rgb`. Privileged simulator signals are rejected
at the schema boundary.

