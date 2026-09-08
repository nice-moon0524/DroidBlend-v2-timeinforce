import os

from core.model_loading import DEFAULT_RECEIVER_PATH, DEFAULT_SENDER_PATH, resolve_model_path


def test_resolve_model_path_prefers_env_var():
    sender_key = "DROID_SPEAK_TEST_SENDER"
    receiver_key = "DROID_SPEAK_TEST_RECEIVER"
    old_sender = os.environ.get(sender_key)
    old_receiver = os.environ.get(receiver_key)
    try:
        os.environ[sender_key] = "D:\\models\\sender"
        os.environ[receiver_key] = "D:\\models\\receiver"
        assert resolve_model_path(sender_key, DEFAULT_SENDER_PATH) == "D:\\models\\sender"
        assert resolve_model_path(receiver_key, DEFAULT_RECEIVER_PATH) == "D:\\models\\receiver"
        assert DEFAULT_SENDER_PATH == "/opt/hhy/models/Mistral-7B-v0.1"
        assert DEFAULT_RECEIVER_PATH == "/opt/hhy/models/mistrallite"
    finally:
        if old_sender is None:
            os.environ.pop(sender_key, None)
        else:
            os.environ[sender_key] = old_sender
        if old_receiver is None:
            os.environ.pop(receiver_key, None)
        else:
            os.environ[receiver_key] = old_receiver
