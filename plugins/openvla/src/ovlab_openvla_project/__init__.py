"""Locate this plugin's installed manifest without importing a model runtime."""
from importlib.metadata import distribution

PLUGIN_VERSION = "0.2.0"


def project_root():
    package = distribution("ovlab-openvla-plugin")
    for item in package.files or ():
        if item.as_posix().endswith("share/ovlab/plugins/openvla/plugin.json"):
            candidate = package.locate_file(item).resolve().parent
            if (candidate / "plugin.json").is_file():
                return candidate
    # pip --target relocates data_files into the target while RECORD retains
    # the interpreter installation scheme's relative share/ path.
    candidate = package.locate_file("share/ovlab/plugins/openvla").resolve()
    if (candidate / "plugin.json").is_file():
        return candidate
    raise FileNotFoundError("the installed OpenVLA plugin distribution has no project manifest")
