"""Read-only interpretation of artifact fields written before typed extensions.

New runs record explicit benchmark identity, suite and renderer fields. These
fallbacks only interpret immutable historical artifacts; they never rewrite them.
"""
def renderer_metadata(runtime):
    return runtime.get('renderer', runtime.get('libero_renderer'))


def benchmark_family(context, started):
    explicit = started.get('metadata', {}).get('benchmark_identity', {})
    if explicit:
        return explicit.get('metadata', {}).get('family', explicit.get('benchmark_id'))
    suite = context.get('suite_name')
    if isinstance(suite, str) and suite.upper().startswith('LIBERO'):
        return 'LIBERO'
    return started.get('benchmark', {}).get('name')


def legacy_suite(tasks):
    for task in tasks:
        task_id = str(task['task_id'])
        for prefix, suite in (('libero/10/', 'libero_10'), ('libero/spatial/', 'libero_spatial'),
                              ('libero/object/', 'libero_object'), ('libero/goal/', 'libero_goal')):
            if task_id.startswith(prefix): return suite
    return None
