# OVLAB Remote Policy

`ovlab-remote-policy` provides the generic, versioned local RPC boundary used
when an OVLAB runner and a policy require incompatible Python environments.
The transport is an `AF_UNIX` socket with length-prefixed JSON frames. Policy
services remain ordinary `PolicyAdapter` implementations behind the boundary;
the runner uses `RemotePolicyAdapter` and contains no model-specific logic.

The prediction schema deliberately carries only identifiers, the authoritative
instruction, and `camera.primary.rgb`. Privileged simulator signals are rejected
at the schema boundary.

