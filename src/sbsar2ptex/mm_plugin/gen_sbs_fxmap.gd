@tool
extends MMGenBase


# From the Material Maker fork (rbroderi/material-maker-mod), under its MIT
# licence; installed with the sbs plugin, whose plugin.gd registers it (no
# class_name: scripts in user:// are not in Godot's class cache).

# FX-Map generator (as Substance Designer's FX-Map node): splats of patterns
# (built-in shapes or the node's pattern inputs, sampled live) are drawn with
# a blending mode, in order, on a background.
#
# The splats are generated on the GPU by walk_code, GLSL that is run for each
# of the "invocations" (e.g. 1 invocation that walks a whole quadrant tree with
# a sequential random generator, or 1 per independent subtree) and calls
#	fx_emit(vec2 position, vec2 size, float rotation, vec4 color, int pattern,
#	        int blend, float opacity, float pattern_parameter)
# fx_invocation is the invocation's index, fx_seed the node's seed, and fx_size
# the output size in pixels. $name in the code is the node's parameter name
# (from fx_parameters) or a named parameter of the graph. global_code is
# added before the walk (e.g. the FX-Map random generator).
#
# Splats are then binned in a grid of cells (on the CPU, in emission order),
# and drawn by a gather shader: each pixel draws the splats of its cell, in
# order. Positions are in UV space (0..1, wrapped: splats crossing the borders
# are drawn on both sides), sizes are relative to the output, rotation is in
# turns. Patterns: see PATTERN_CODE, patterns 100+k sample pattern input k,
# 200+ call fx_custom_pattern() if global_code defines FX_CUSTOM_PATTERNS.
# Blending modes: 0 add, 1 subtract, 2 max, 3 min, 4 alpha, 5 copy.


const MAX_PATTERN_INPUTS : int = 4
const FLOATS_PER_SPLAT : int = 16

const WALK_TEMPLATE : String = """#version 450

layout(local_size_x = @LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

@DECLARATIONS

layout(set = 3, binding = 0, std140) restrict buffer MM {
	int mm_chunk_y;
};

const float PI = 3.14159265359;
const float TAU = 2.0*PI;

@COMMON_SHADER_FUNCTIONS

@GLOBALS

uint fx_invocation;
uint fx_count;

void fx_emit(vec2 position, vec2 size, float rotation, vec4 color, int pattern, int blend, float opacity, float pattern_parameter) {
	if (fx_count < uint(fx_max_splats)) {
		int i = (int(fx_invocation)*fx_max_splats+int(fx_count))*16;
		mm_out.fx_splats[i] = position.x;
		mm_out.fx_splats[i+1] = position.y;
		mm_out.fx_splats[i+2] = size.x;
		mm_out.fx_splats[i+3] = size.y;
		mm_out.fx_splats[i+4] = rotation;
		mm_out.fx_splats[i+5] = float(pattern);
		mm_out.fx_splats[i+6] = float(blend);
		mm_out.fx_splats[i+7] = opacity;
		mm_out.fx_splats[i+8] = color.r;
		mm_out.fx_splats[i+9] = color.g;
		mm_out.fx_splats[i+10] = color.b;
		mm_out.fx_splats[i+11] = color.a;
		mm_out.fx_splats[i+12] = pattern_parameter;
	}
	fx_count += 1u;
}

@DEFINITIONS

void main() {
	fx_invocation = gl_GlobalInvocationID.x;
	if (fx_invocation >= uint(fx_invocations)) {
		return;
	}
	fx_count = 0u;
	fx_walk();
	mm_out.fx_counts[fx_invocation] = int(min(fx_count, uint(fx_max_splats)));
}
"""

const GATHER_TEMPLATE : String = """#version 450

layout(local_size_x = @LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

@DECLARATIONS

layout(set = 3, binding = 0, std140) restrict buffer MM {
	int mm_chunk_y;
};

const float PI = 3.14159265359;
const float TAU = 2.0*PI;

@COMMON_SHADER_FUNCTIONS

@GLOBALS

@DEFINITIONS

void main() {
	vec2 pixel = gl_GlobalInvocationID.xy+vec2(0.5, 0.5+mm_chunk_y);
	vec2 image_size = imageSize(OUTPUT_TEXTURE);
	vec2 uv = pixel/image_size;
	@CODE
	vec4 color = @OUTPUT_VALUE;
	ivec2 cell = clamp(ivec2(uv*float(fx_cells)), ivec2(0), ivec2(fx_cells-1));
	int cell_index = cell.y*fx_cells+cell.x;
	int first = mm_out.fx_cell_starts[cell_index];
	int last = mm_out.fx_cell_starts[cell_index+1];
	for (int e = first; e < last; ++e) {
		int entry = mm_out.fx_cell_entries[e];
		int i = (entry >> 4)*16;
		vec2 offset = vec2(float((entry & 3)-1), float(((entry >> 2) & 3)-1));
		vec2 position = vec2(mm_out.fx_splats[i], mm_out.fx_splats[i+1])+offset;
		vec2 size = vec2(mm_out.fx_splats[i+2], mm_out.fx_splats[i+3]);
		float angle = -TAU*mm_out.fx_splats[i+4];
		vec2 d = uv-position;
		d = mat2(vec2(cos(angle), sin(angle)), vec2(-sin(angle), cos(angle)))*d;
		vec2 p = d/max(abs(size), vec2(1e-8))*sign(size+vec2(1e-12));
		if (abs(p.x) > 0.5 || abs(p.y) > 0.5) {
			continue;
		}
		vec4 splat_color = vec4(mm_out.fx_splats[i+8], mm_out.fx_splats[i+9], mm_out.fx_splats[i+10], mm_out.fx_splats[i+11]);
		vec4 pattern = fx_pattern(int(mm_out.fx_splats[i+5]), p, mm_out.fx_splats[i+12])*splat_color;
		color = fx_blend(color, pattern, int(mm_out.fx_splats[i+6]), mm_out.fx_splats[i+7]);
	}
	imageStore(OUTPUT_TEXTURE, ivec2(pixel), color);
}
"""

## Built-in patterns (p in -0.5..0.5) and blending modes
const PATTERN_CODE : String = """
vec4 fx_custom_pattern(int pattern, vec2 p, float parameter);

vec4 fx_pattern(int pattern, vec2 p, float parameter) {
	vec2 q = p*2.0;
	float r = length(q);
	float v = 0.0;
	if (pattern >= 200) {
#ifdef FX_CUSTOM_PATTERNS
		return fx_custom_pattern(pattern, p, parameter);
#else
		return vec4(0.0);
#endif
	}
	@PATTERN_INPUTS
	switch (pattern) {
		case 0: return vec4(0.0);
		case 1: v = 1.0; break;                                                  // square
		case 2: v = step(r, 1.0); break;                                         // disc
		case 3: v = max(1.0-r*r, 0.0); break;                                    // paraboloid
		case 4: v = r < 1.0 ? 0.5+0.5*cos(PI*r) : 0.0; break;                    // bell
		case 5: v = exp(-r*r*4.0); break;                                        // gaussian
		case 6: v = max(1.0-r, 0.0)*max(1.0-r, 0.0); break;                      // thorn
		case 7: v = 1.0-max(abs(q.x), abs(q.y)); break;                          // pyramid
		case 8: v = step(abs(q.y), 0.5); break;                                  // brick
		case 9: v = p.x+0.5; break;                                              // gradation
		case 10: v = 0.5+0.5*cos(TAU*max(parameter, 1.0)*q.x); break;            // waves
		case 11: v = q.y > 0.0 ? 0.0 : (r < 1.0 ? 0.5+0.5*cos(PI*r) : 0.0); break; // half bell
		case 12: v = r < 1.0 ? abs(cos(PI*r*max(parameter, 1.0))) * (1.0-r) : 0.0; break; // ridged bell
		case 13: v = step(r, 1.0)*step(1.0, length(q-vec2(0.5, 0.0))*2.0/1.5); break; // crescent
		case 14: v = step(length(vec2(max(abs(q.x)-0.5, 0.0), q.y)), 0.5); break; // capsule
		case 15: v = max(1.0-r, 0.0); break;                                     // cone
	}
	return vec4(vec3(clamp(v, 0.0, 1.0)), 1.0);
}

vec4 fx_blend(vec4 dst, vec4 src, int blend, float opacity) {
	switch (blend) {
		case 0: return dst+src*opacity;
		case 1: return dst-src*opacity;
		case 2: return mix(dst, max(dst, src), opacity);
		case 3: return mix(dst, min(dst, src), opacity);
		case 4: return mix(dst, src, src.a*opacity);
		case 5: return src;
	}
	return dst;
}
"""

var shader_walk : MMShaderCompute = MMShaderCompute.new()
var shader_gather : MMShaderCompute = MMShaderCompute.new()
## Shaders that render the pattern inputs (and the background) to textures
var input_shaders : Array[MMShaderCompute] = []
var input_textures : Array[MMTexture] = []
var texture : MMTexture = MMTexture.new()
var exiting : bool = false
var used_named_parameters : Array = []
var splat_count : int = 0


func _init() -> void:
	for i in MAX_PATTERN_INPUTS+1:
		input_shaders.append(MMShaderCompute.new())
		input_textures.append(MMTexture.new())

func _ready() -> void:
	super._ready()
	mm_deps.create_buffer(get_buffer_name(), self)
	for i in get_pattern_input_count()+1:
		mm_deps.create_buffer(get_input_buffer_name(i), self)
	update_shaders()

func _enter_tree() -> void:
	exiting = false

func _exit_tree() -> void:
	exiting = true

func get_type() -> String:
	return "sbs_fxmap"

func get_type_name() -> String:
	return "FX-Map (Substance)"

func get_description() -> String:
	return "Splats patterns (built-in shapes or the pattern inputs) generated by GPU code"

func get_buffer_name() -> String:
	return "o%d_tex" % get_instance_id()

## Input 0 is the background, inputs 1.. are the patterns
func get_input_buffer_name(i : int) -> String:
	return "o%d_fx_in%d" % [ get_instance_id(), i ]

func get_pattern_input_count() -> int:
	return clampi(int(parameters.get("pattern_inputs", 0)), 0, MAX_PATTERN_INPUTS)

func get_buffers(_flags : int = BUFFERS_ALL) -> Array:
	return [ self ]

## Parameters of the walk code, declared in the node's data:
## [ { name, label, type ("float", "color" or "boolean"), min, max, step, default } ]
var fx_parameters : Array = []

func get_parameter_defs() -> Array:
	var rv : Array = [
		{ name="size_mode", label="Size mode", type="enum", default=1,
			values=[ { name="Absolute", value="absolute" }, { name="Relative to output", value="relative" }, { name="Relative to input", value="input" } ] },
		{ name="size", label="Size", type="size", first=4, last=14, default=9 },
		{ name="size_offset", label="Size offset", type="float", min=-6.0, max=4.0, step=1.0, default=0.0 },
		{ name="format", label="Format", type="enum", default=0, values=MMGenIterateBufferV2.FORMATS },
		{ name="grayscale", label="Grayscale", type="boolean", default=false },
		{ name="pattern_inputs", label="Pattern inputs", type="float", min=0, max=MAX_PATTERN_INPUTS, step=1, default=0 },
		{ name="pattern_size", label="Pattern size", type="size", first=4, last=12, default=8 },
		{ name="background", label="Background", type="color", default={ r=0.0, g=0.0, b=0.0, a=1.0 } },
		{ name="invocations", label="Invocations", type="float", min=1, max=4096, step=1, default=1 },
		{ name="max_splats", label="Splats per invocation", type="float", min=1, max=1048576, step=1, default=4096 },
		{ name="cells", label="Cells", type="float", min=1, max=256, step=1, default=32 },
	]
	for p in fx_parameters:
		rv.append(p)
	return rv

func get_input_defs() -> Array:
	var rv : Array = [ { name="background", label="Background", type="rgba" } ]
	for i in get_pattern_input_count():
		rv.append({ name="pattern%d" % i, label="Pattern %d" % (i+1), type="rgba" })
	return rv

func get_output_defs(_show_hidden : bool = false) -> Array:
	return [ { type="rgba" } ]

func source_changed(_input_index : int) -> void:
	update_shaders()

func all_sources_changed() -> void:
	update_shaders()

func set_parameter(n : String, v) -> void:
	super.set_parameter(n, v)
	if not is_inside_tree():
		return
	if n == "pattern_inputs":
		for i in get_pattern_input_count()+1:
			mm_deps.create_buffer(get_input_buffer_name(i), self)
		emit_signal("parameter_changed", "__update_all__", null)
	if n in [ "pattern_inputs", "pattern_size", "format", "grayscale", "max_splats", "invocations", "walk_code", "global_code", "cells", "background" ]:
		update_shaders()
	else:
		mm_deps.buffer_invalidate(get_buffer_name())

func get_output_dimensions() -> Vector2i:
	var size : int = int(get_parameter("size"))
	match int(get_parameter("size_mode")):
		MMGenBuffer.SIZE_MODE_RELATIVE_TO_OUTPUT:
			var material : Node = find_material()
			if material != null and material.parameters.has("size"):
				size = int(material.parameters.size)+roundi(float(get_parameter("size_offset")))
		MMGenBuffer.SIZE_MODE_RELATIVE_TO_INPUT:
			var width : int = MMGenBuffer.find_texture_width(get_source(0))
			if width > 0:
				size = roundi(log(float(width))/log(2.0))+roundi(float(get_parameter("size_offset")))
	size = clampi(size, 0, 14)
	return Vector2i(1 << size, 1 << size)

func get_buffer_dimensions() -> Vector2i:
	return get_output_dimensions()

## Replaces $name by the uniforms of the node's parameters and of the graph's
## named parameters
func replace_parameters(code : String, rv : ShaderCode) -> String:
	var genname : String = "o%d" % get_instance_id()
	var names : Array = []
	for p in fx_parameters:
		names.append(p.name)
	# Longer names first ($ab before $a)
	names.sort_custom(func(a, b): return a.length() > b.length())
	for n in names:
		if code.contains("$"+n):
			var def : Dictionary = get_parameter_def(n)
			var uniform_name : String = "p_%s_%s" % [ genname, n ]
			var value = get_parameter(n)
			match def.get("type", "float"):
				"color":
					rv.add_uniform(uniform_name, "vec4", MMType.deserialize_value(value) if value is Dictionary else value)
				"boolean":
					rv.add_uniform(uniform_name, "float", 1.0 if value else 0.0)
				_:
					rv.add_uniform(uniform_name, "float", float(value) if not value is String else 0.0)
			code = code.replace("$"+n, uniform_name)
	var parent : Node = get_parent()
	if parent != null and parent.has_method("get_named_parameters"):
		var named_parameters : Dictionary = parent.get_named_parameters()
		var named : Array = named_parameters.keys()
		named.sort_custom(func(a, b): return a.length() > b.length())
		for np in named:
			if code.contains("$"+np):
				code = code.replace("$"+np, named_parameters[np].id)
				rv.add_uniform(named_parameters[np].id, "float", named_parameters[np].value)
	return code

var updating_shaders : bool = false

func update_shaders() -> void:
	if not updating_shaders:
		updating_shaders = true
		do_update_shaders.call_deferred()

func do_update_shaders() -> void:
	updating_shaders = false
	if not is_inside_tree() or exiting:
		return
	var genname : String = "o%d" % get_instance_id()
	var max_splats : int = max(1, int(get_parameter("max_splats")))
	var invocations : int = max(1, int(get_parameter("invocations")))
	var cells : int = clampi(int(get_parameter("cells")), 1, 256)
	# Inputs: background and patterns, rendered to textures
	for i in get_pattern_input_count()+1:
		var source_output : OutputPort = get_source(i)
		var source : ShaderCode
		if source_output != null:
			source = source_output.generator.get_shader_code("uv", source_output.output_index, MMGenContext.new())
		if source == null or source.output_type == "":
			source = ShaderCode.new()
			source.output_type = "rgba"
			var background : Color = MMType.deserialize_value(get_parameter("background")) if get_parameter("background") is Dictionary else get_parameter("background")
			source.output_values.rgba = "vec4(%.9f, %.9f, %.9f, %.9f)" % [ background.r, background.g, background.b, background.a ] if i == 0 else "vec4(0.0)"
		if await input_shaders[i].set_shader_from_shadercode_storage(source, MMPipeline.STORAGE_16F):
			await mm_deps.buffer_create_compute_material(get_input_buffer_name(i), input_shaders[i])
	# Walk
	var walk : ShaderCode = ShaderCode.new()
	walk.output_type = "rgba"
	walk.add_uniform("fx_seed", "float", get_seed())
	walk.add_uniform("fx_size", "vec2", Vector2(get_output_dimensions()))
	walk.add_uniform("fx_invocations", "int", invocations)
	walk.add_uniform("fx_max_splats", "int", max_splats)
	var global_code : String = replace_parameters(str(parameters.get("global_code", "")), walk)
	var walk_code : String = replace_parameters(str(parameters.get("walk_code", "")), walk)
	# Preprocessor lines (e.g. #define FX_CUSTOM_PATTERNS) must stay in order
	walk.defs = global_code+"\nvoid fx_walk() {\n%s\n}\n" % walk_code
	walk.output_values.rgba = ""
	var walk_outputs : Array[Dictionary] = [
		{ name="fx_splats", type="float", array_size=invocations*max_splats*FLOATS_PER_SPLAT },
		{ name="fx_counts", type="int", array_size=invocations } ]
	shader_walk = MMShaderCompute.new()
	var walk_parameters : Array[Dictionary] = [
		{ name="fx_invocations", type="int", value=invocations },
		{ name="fx_max_splats", type="int", value=max_splats } ]
	if not await shader_walk.compute_shader.set_shader_from_shadercode_ext(WALK_TEMPLATE, walk, [], walk_parameters, false, walk_outputs):
		print("FX-Map %s: the walk code does not compile" % get_hier_name())
		return
	# Gather
	var gather : ShaderCode = ShaderCode.new()
	gather.output_type = "rgba"
	gather.add_uniform("fx_cells", "int", cells)
	var pattern_inputs : String = ""
	for i in get_pattern_input_count()+1:
		gather.add_uniform("%s_fx_in%d" % [ genname, i ], "sampler2D", input_textures[i])
		if i > 0:
			pattern_inputs += "if (pattern == %d) { return textureLod(%s_fx_in%d, p+vec2(0.5), 0.0); }\n" % [ 99+i, genname, i ]
	gather.defs = global_code+"\n"+PATTERN_CODE.replace("@PATTERN_INPUTS", pattern_inputs)
	gather.output_values.rgba = "textureLod(%s_fx_in0, uv, 0.0)" % genname
	var gather_outputs : Array[Dictionary] = [
		{ name="fx_splats", type="float", array_size=invocations*max_splats*FLOATS_PER_SPLAT },
		{ name="fx_cell_starts", type="int", array_size=cells*cells+1 },
		{ name="fx_cell_entries", type="int", array_size=get_max_cell_entries() } ]
	shader_gather = MMShaderCompute.new()
	var channels : int = 1 if get_parameter("grayscale") else 4
	var storage : int = [ MMPipeline.STORAGE_16F, MMPipeline.STORAGE_32F, MMPipeline.STORAGE_8, MMPipeline.STORAGE_16 ][clampi(int(get_parameter("format")), 0, 3)]
	var output_textures : Array[Dictionary] = [ { name="OUTPUT_TEXTURE", type=MMPipeline.get_texture_type(channels, storage) } ]
	var gather_parameters : Array[Dictionary] = [ { name="fx_cells", type="int", value=cells } ]
	if not await shader_gather.compute_shader.set_shader_from_shadercode_ext(GATHER_TEMPLATE, gather, output_textures, gather_parameters, false, gather_outputs):
		print("FX-Map %s: the gather shader does not compile" % get_hier_name())
		return
	# The node's buffer depends on the parameters of both shaders
	await mm_deps.buffer_create_compute_material(get_buffer_name(), shader_walk)
	for p in shader_gather.get_parameters().keys():
		var value = mm_deps.buffer_add_dependency(get_buffer_name(), p)
		if value != null:
			shader_gather.set_parameter(p, value)
	mm_deps.buffer_invalidate(get_buffer_name())

## Size of the cells' splat lists (splats can cover several cells)
func get_max_cell_entries() -> int:
	return max(1, int(get_parameter("invocations")))*max(1, int(get_parameter("max_splats")))*16

func on_dep_update_value(buffer_name, parameter_name, value) -> bool:
	if value == null:
		return false
	if buffer_name == get_buffer_name():
		shader_walk.compute_shader.set_parameter(parameter_name, value, true)
		shader_gather.compute_shader.set_parameter(parameter_name, value, true)
	else:
		for i in get_pattern_input_count()+1:
			if buffer_name == get_input_buffer_name(i):
				input_shaders[i].set_parameter(parameter_name, value)
	return false

func on_dep_update_buffer(buffer_name : String) -> bool:
	for i in get_pattern_input_count()+1:
		if buffer_name == get_input_buffer_name(i):
			var pattern_size : int = 1 << int(get_parameter("pattern_size"))
			var dimensions : Vector2i = get_output_dimensions() if i == 0 else Vector2i(pattern_size, pattern_size)
			var status : bool = await input_shaders[i].compute_shader.render(input_textures[i], dimensions)
			if status:
				mm_deps.dependency_update(buffer_name, input_textures[i], true)
			return status
	if buffer_name != get_buffer_name():
		return false
	var start_time : int = Time.get_ticks_msec()
	var max_splats : int = max(1, int(get_parameter("max_splats")))
	var invocations : int = max(1, int(get_parameter("invocations")))
	var cells : int = clampi(int(get_parameter("cells")), 1, 256)
	# Walk: generate the splats
	var walk_values : Dictionary = {}
	var splats : PackedFloat32Array = PackedFloat32Array()
	splats.resize(invocations*max_splats*FLOATS_PER_SPLAT)
	walk_values.fx_splats = splats
	var counts : PackedInt32Array = PackedInt32Array()
	counts.resize(invocations)
	walk_values.fx_counts = counts
	if not await shader_walk.compute_shader.render_ext([], Vector2i(invocations, 1), walk_values):
		return false
	splats = walk_values.fx_splats
	counts = walk_values.fx_counts
	# Bin: lists of splats by cell, in emission order
	var bins : Array[PackedInt32Array] = []
	bins.resize(cells*cells)
	splat_count = 0
	for invocation in invocations:
		for s in counts[invocation]:
			var index : int = invocation*max_splats+s
			var o : int = index*FLOATS_PER_SPLAT
			splat_count += 1
			var position : Vector2 = Vector2(splats[o], splats[o+1])
			# Bounding radius of the rotated splat
			var radius : float = 0.5*Vector2(splats[o+2], splats[o+3]).length()
			var min_cell : Vector2i = Vector2i(floori((position.x-radius)*cells), floori((position.y-radius)*cells))
			var max_cell : Vector2i = Vector2i(floori((position.x+radius)*cells), floori((position.y+radius)*cells))
			# Cells covered (wrapped), with the offset of the splat's copy
			var added : Dictionary = {}
			for cy in range(min_cell.y, max_cell.y+1):
				for cx in range(min_cell.x, max_cell.x+1):
					var wx : int = posmod(cx, cells)
					var wy : int = posmod(cy, cells)
					# The copy of the splat drawn in this cell is moved by a whole tile
					var offset_x : int = (wx-cx)/cells
					var offset_y : int = (wy-cy)/cells
					if abs(offset_x) > 1 or abs(offset_y) > 1:
						continue
					var entry : int = (index << 4) | ((offset_y+1) << 2) | (offset_x+1)
					var key : int = (wy*cells+wx)*16+((offset_y+1) << 2)+(offset_x+1)
					if added.has(key):
						continue
					added[key] = true
					bins[wy*cells+wx].append(entry)
	var starts : PackedInt32Array = PackedInt32Array()
	starts.resize(cells*cells+1)
	var entries : PackedInt32Array = PackedInt32Array()
	for c in cells*cells:
		starts[c] = entries.size()
		entries.append_array(bins[c])
	starts[cells*cells] = entries.size()
	var max_entries : int = get_max_cell_entries()
	if entries.size() > max_entries:
		push_warning("FX-Map %s: too many splats per cell, some are not drawn" % get_hier_name())
		entries.resize(max_entries)
		for c in starts.size():
			starts[c] = min(starts[c], max_entries)
	entries.resize(max_entries)
	# Gather: draw the splats
	var gather_values : Dictionary = { fx_splats=splats, fx_cell_starts=starts, fx_cell_entries=entries }
	var status : bool = await shader_gather.compute_shader.render_ext([ texture ], get_output_dimensions(), gather_values)
	rendering_time = Time.get_ticks_msec()-start_time
	rendering_time_updated.emit(rendering_time)
	if status:
		mm_deps.dependency_update(buffer_name, texture, true)
	return status

func _get_shader_code(uv : String, _output_index : int, context : MMGenContext) -> ShaderCode:
	var genname : String = "o%d" % get_instance_id()
	var rv : ShaderCode = ShaderCode.new()
	var grayscale : bool = get_parameter("grayscale")
	rv.output_type = "f" if grayscale else "rgba"
	var variant_index : int = context.get_variant(self, uv)
	if variant_index == -1:
		variant_index = context.get_variant(self, uv)
		rv.add_uniform(get_buffer_name(), "sampler2D", texture)
		rv.code = "%s %s_%d = textureLod(%s, fract(%s), 0.0)%s;\n" % [ "float" if grayscale else "vec4", genname, variant_index, get_buffer_name(), uv, ".r" if grayscale else "" ]
	rv.output_values[rv.output_type] = "%s_%d" % [ genname, variant_index ]
	return rv

func get_output_attributes(_output_index : int) -> Dictionary:
	return { texture=get_buffer_name() }

func _serialize(data : Dictionary) -> Dictionary:
	data.type = get_type()
	data.fx_parameters = fx_parameters.duplicate(true)
	for k in [ "walk_code", "global_code" ]:
		data.parameters[k] = parameters.get(k, "")
	return data

func _deserialize(data : Dictionary) -> void:
	if data.get("fx_parameters") is Array:
		fx_parameters = data.fx_parameters.duplicate(true)
	if data.has("parameters"):
		for k in [ "walk_code", "global_code" ]:
			if data.parameters.has(k):
				parameters[k] = data.parameters[k]
