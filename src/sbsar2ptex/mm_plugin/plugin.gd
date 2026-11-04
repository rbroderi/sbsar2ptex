extends MMPlugin


# The sbs plugin's script: node types implemented by scripts (Substance's
# FX-Map, sbs_fxmap, and Substance's way of drawing its patterns, sbs_splats);
# the atomic nodes are .mmg files in nodes/.


func _enable(api : MMPlugins.API) -> void:
	api.add_node_type("sbs_fxmap", preload("gen_sbs_fxmap.gd"))
	api.add_node_type("sbs_splats", preload("gen_sbs_splats.gd"))
