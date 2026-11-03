extends MMPlugin


# The sbs plugin's script: node types implemented by scripts (Substance's
# FX-Map, sbs_fxmap); the atomic nodes are .mmg files in nodes/.


func _enable(api : MMPlugins.API) -> void:
	api.add_node_type("sbs_fxmap", preload("gen_sbs_fxmap.gd"))
