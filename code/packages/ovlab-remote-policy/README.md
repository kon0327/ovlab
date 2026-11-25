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

The v2 prediction schema carries the authoritative instruction, timestamps and
named image/proprioception arrays from the negotiated OVLAB observation contract.
No camera name, image count, RGB shape or float32 proprioception is imposed by
the transport. Zero images and zero proprioception vectors are supported.
Array dtypes, shapes, encodings, color spaces, camera names and per-input
sample timestamps survive the boundary. Frames and arrays retain bounded sizes;
object/string/complex dtypes and arbitrary observation metadata are rejected.
Privileged simulator signals are never prediction fields.

Both worker images must use RPC `ovlab-policy-rpc/2.0.0`. Rebuild policy and
benchmark images together after this protocol change; mixed versions fail at
frame validation before prediction.
