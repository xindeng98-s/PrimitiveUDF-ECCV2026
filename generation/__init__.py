"""V3 unsigned-distance surface extraction."""
GENERATION_TYPES = ('auto', 'udf')
def resolve_generation_type(cfg, field_type='auto'):
    if field_type not in GENERATION_TYPES or cfg.get('is_sdf', False):
        raise ValueError('The V3 release generates unsigned-distance surfaces')
    return 'udf'
def build_mesh_generator(model, cfg, *, device, field_type='auto', **kwargs):
    resolve_generation_type(cfg, field_type)
    from .mesh_udf import MeshUDFGenerator
    return MeshUDFGenerator(model, cfg, device=device, **kwargs)
