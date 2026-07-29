from app.plugins.runner import run_from_config

print('Starting pipeline run...')
result = run_from_config('config_pipeline_full.yaml')
print('Pipeline completed.')
print(result['translation_result'])
