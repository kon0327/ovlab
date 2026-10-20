"""Persistent model endpoints preserve RPC inference across client sessions."""
import socket
import threading
import time

from conftest import SevenDPolicy, identity_provider
from ovlab_remote_policy import TcpPolicyClient
from ovlab_remote_policy.service import PolicyService


def test_tcp_service_keeps_loaded_model_between_sessions(tmp_path, contexts, observation):
    run, episode = contexts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    adapter = SevenDPolicy()
    service = PolicyService(tmp_path / "unused.sock", adapter, identity_provider=identity_provider,
                            startup_context=run, listen_address=("127.0.0.1", port), persistent=True)
    thread = threading.Thread(target=service.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while True:
        client = TcpPolicyClient("127.0.0.1", port, request_timeout_s=2)
        try:
            client.connect()
            break
        except ConnectionRefusedError:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    try:
        assert client.health()["state"] == "ready"
        client.initialize(run)
        client.reset_episode(episode)
        assert client.predict(observation)["actions"].shape == (1, 7)
        health = TcpPolicyClient("127.0.0.1", port, request_timeout_s=2)
        health.connect()
        assert health.health()["state"] == "episode_active"
        health.close_socket()
        # A client closing mid-episode must release its session, not the model.
        client.request_close()
        client.close_socket()
        second = TcpPolicyClient("127.0.0.1", port)
        second.connect()
        assert second.health()["state"] == "ready"
        second.initialize(run)
        second.reset_episode(episode)
        second.predict(observation)
        second.end_episode(episode)
        # Test teardown requests the ordinary shutdown path.
        service.persistent = False
        second.request_close()
        second.close_socket()
    finally:
        client.close_socket()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not (tmp_path / "unused.sock").exists()
