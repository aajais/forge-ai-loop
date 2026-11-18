"""One real SFT-continuation update using newly captured exact-token rollouts."""
import asyncio
import json
from pathlib import Path

from bufogen.training.rl import RLTrainer
from bufogen.training.rl import require_rl_runtime

async def main():
    name="bufo-gemma-dynamo-rl-20261015"
    output=Path("runs")/name
    if output.exists():
        raise RuntimeError("Existing attempt; inspect state and remote job before any new submission")
    state=json.loads(Path("runs/forge-loop/forge-state.json").read_text())
    tasks=json.loads(Path("runs/forge-loop/rl-environment-artifact.json").read_text())
    print(json.dumps({"runtime":require_rl_runtime(),"initial_checkpoint":state["model_artifact_ref"],"requested_updates":1}),flush=True)
    trainer=RLTrainer(tasks["reference"],name=name,base_model=state["base_model"],
        calibration_path=Path("runs/forge-loop/calibration/rl-calibration.json"),
        initial_checkpoint=state["model_artifact_ref"],forge_finetune_id=state["finetune_id"],
        forge_project_alias=state["project_alias"])
    ref=await trainer.train(steps=1)
    result={"model_artifact_ref":ref,"training_record":str(output/"rl-training.json"),"steps_requested":1,
            "scope":"Service-compatibility proof from the original SFT checkpoint; quality improvement not yet measured."}
    Path("runs/forge-gap/rl-result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    asyncio.run(main())
