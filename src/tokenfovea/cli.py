from lmms_eval import models
from lmms_eval.cli.dispatch import main as lmms_main
from lmms_eval.models import ModelManifest


def main() -> None:
    for manifest in (
        ModelManifest(
            model_id="tokenfovea_qwen2_5_vl",
            simple_class_path="tokenfovea.integrations.lmms_eval.TokenFoveaQwen25VL",
        ),
        ModelManifest(
            model_id="tokenfovea_qwen3_5",
            simple_class_path="tokenfovea.integrations.lmms_eval.TokenFoveaQwen35",
        ),
    ):
        models.MODEL_REGISTRY_V2.register_manifest(manifest, overwrite=True)
        models.AVAILABLE_SIMPLE_MODELS[manifest.model_id] = manifest.simple_class_path
        models.AVAILABLE_MODELS[manifest.model_id] = manifest.simple_class_path.rsplit(".", 1)[-1]
    lmms_main()
