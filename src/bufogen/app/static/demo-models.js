// Shared ordering and roles for the presenter-facing model choices.
const DEMO_MODELS = [
  {id: 'qwen_base', role: 'Base'},
  {id: 'sft_q3', role: 'SFT · 1 epoch'},
  {id: 'rl_q3', role: 'SFT + RL'},
  {id: 'closed_quality_bar', role: 'Teacher'},
];
function demoModels(models) {
  return DEMO_MODELS.map(choice => {
    const model = models.find(model => model.id === choice.id && !model.routing_alias && model.model);
    return model ? {...model, role: choice.role} : null;
  }).filter(Boolean);
}
