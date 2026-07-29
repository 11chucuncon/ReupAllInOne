from app.plugins import runner
cfg = runner.load_plugin_config('config_pipeline_full.yaml')
pl = runner.build_pipeline_from_config(cfg)
print('Loaded config steps:', [s.__class__.__name__ for s in pl.steps])
