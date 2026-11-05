@tool
extends MMGenBase


# Installed with the sbs plugin, whose plugin.gd registers it (no class_name:
# scripts in user:// are not in Godot's class cache).

# Splats (Substance): an FX-Map's patterns drawn as Substance Designer's GPU
# engine draws them, for graphs converted by sbsar2ptex. Each pattern is one
# triangle over its bounding box, its corners snapped to the pixel grid, the
# position in the pattern (q, -1 to 1 across it) set at the corners and
# interpolated by the rasterizer: a pixel's q is the GPU's interpolation, a
# float32 step off a per-pixel computation here and there, as Substance's.
#
# The splats (parameter "splats", written by sbsar2ptex: little-endian
# float32s in base64, as arrays would be read as gradients) are
# FLOATS_PER_SPLAT floats each, drawn in order: the color (straight), the
# bounding box's
# size and center, the pattern's size, rotation (turns) and supplement, the
# pattern's id and whether it blends by max (else it adds). Tile copies are
# splats of their own. Patterns are "pattern_code"'s
#	bool fxpat_fx(int id, vec2 q, float s, out float v)
# (false where the pattern does not cover q); patterns from 15 add their
# supplement after the color. The output (rgba, 32-bit floats) starts at the
# background. The image is the render region: its left, top, width and height
# (parameters region_x, region_y, region_w, region_h; the unit square's top is
# 1), in the splats' units.


const FLOATS_PER_SPLAT : int = 16

const VERTEX_SHADER : String = """#version 450

struct Splat {
	vec4 colour;
	vec4 frame; // bounding box size, center
	vec4 shape; // pattern size, rotation (turns), supplement
	vec4 kind;  // pattern id, max blending
};

layout(set = 0, binding = 0, std430) restrict readonly buffer Splats {
	Splat splats[];
};

layout(push_constant, std430) uniform Params {
	vec4 view;   // origin, scale to clip space
	vec4 texels; // texels per unit, units per texel
	ivec4 first; // the draw's first splat
} params;

layout(location = 0) out vec2 q;
layout(location = 1) flat out vec4 colour;
layout(location = 2) flat out float suppl;
layout(location = 3) flat out int pattern;

void main() {
	Splat s = splats[params.first.x+gl_InstanceIndex];
	int corner = gl_VertexIndex % 3;
	vec2 c = vec2(corner == 1 ? 3.0 : -1.0, corner == 2 ? 3.0 : -1.0);
	float angle = s.shape.z * 6.2831855;
	float sn = sin(angle);
	float cs = cos(angle);
	vec2 e = c * abs(s.frame.xy);
	vec2 p = s.frame.zw + e * 0.5;
	// The corner snapped to a texel corner, half a texel past it
	vec2 t = (p - params.view.xy) * params.texels.xy - vec2(0.5);
	vec2 o = (((t - floor(t)) - vec2(1.5)) + step(c, vec2(0.0))) * params.texels.zw;
	vec2 g = ((p - o) - params.view.xy) * params.view.zw - vec2(1.0);
	vec2 k = e - o * 2.0;
	vec2 r = k.xx * vec2(cs, -sn) + k.yy * vec2(sn, cs);
	q = vec2(r.x / s.shape.x, r.y / s.shape.y);
	colour = vec4(s.colour.rgb * s.colour.a, s.colour.a);
	suppl = s.shape.w;
	pattern = int(s.kind.x);
	gl_Position = vec4(g.x, -g.y, 0.0, 1.0);
}
"""

const FRAGMENT_SHADER : String = """#version 450

layout(location = 0) in vec2 q;
layout(location = 1) flat in vec4 colour;
layout(location = 2) flat in float suppl;
layout(location = 3) flat in int pattern;
layout(location = 0) out vec4 result;

@PATTERN_CODE

void main() {
	float v;
	if (!fxpat_fx(pattern, q, suppl, v)) {
		discard;
	}
	result = colour * v;
	if (pattern >= 15) {
		result += suppl;
	}
}
"""

var texture : MMTexture = MMTexture.new()
## The shader, compiled for shader_device from shader_patterns
var shader : RID = RID()
var shader_device : RenderingDevice = null
var shader_patterns : String = ""


func _ready() -> void:
	super._ready()
	mm_deps.create_buffer(get_buffer_name(), self)

func _exit_tree() -> void:
	if shader.is_valid() and shader_device != null:
		shader_device.free_rid(shader)
		shader = RID()

func get_type() -> String:
	return "sbs_splats"

func get_type_name() -> String:
	return "Splats (Substance)"

func get_description() -> String:
	return "An FX-Map's patterns, drawn as Substance's GPU engine draws them"

func get_buffer_name() -> String:
	return "o%d_tex" % get_instance_id()

func get_buffers(_flags : int = BUFFERS_ALL) -> Array:
	return [ self ]

func get_parameter_defs() -> Array:
	return [
		{ name="size", label="Size", type="size", first=0, last=14, default=9 },
		{ name="background", label="Background", type="float", min=0.0, max=1.0, step=0.001, default=0.0 },
		{ name="region_x", label="Region left", type="float", min=-16.0, max=16.0, step=0.001, default=0.0 },
		{ name="region_y", label="Region top", type="float", min=-16.0, max=16.0, step=0.001, default=1.0 },
		{ name="region_w", label="Region width", type="float", min=0.001, max=32.0, step=0.001, default=1.0 },
		{ name="region_h", label="Region height", type="float", min=0.001, max=32.0, step=0.001, default=1.0 },
	]

func get_input_defs() -> Array:
	return []

func get_output_defs(_show_hidden : bool = false) -> Array:
	return [ { type="rgba" } ]

func set_parameter(n : String, v) -> void:
	super.set_parameter(n, v)
	if is_inside_tree():
		mm_deps.buffer_invalidate(get_buffer_name())

func get_output_dimensions() -> Vector2i:
	var size : int = 1 << clampi(int(get_parameter("size")), 0, 14)
	return Vector2i(size, size)

func get_buffer_dimensions() -> Vector2i:
	return get_output_dimensions()

func on_dep_update_buffer(buffer_name : String) -> bool:
	if buffer_name != get_buffer_name():
		return false
	var start_time : int = Time.get_ticks_msec()
	await mm_renderer.wait_for_rendering_device()
	var splats : PackedFloat32Array = Marshalls.base64_to_raw(str(parameters.get("splats", ""))).to_float32_array()
	var patterns : String = str(parameters.get("pattern_code", ""))
	var status = await mm_renderer.thread_run(in_thread_render, [ splats, patterns, get_output_dimensions() ])
	rendering_time = Time.get_ticks_msec()-start_time
	rendering_time_updated.emit(rendering_time)
	if status:
		mm_deps.dependency_update(buffer_name, texture, true)
	return status == true

func compile_shader(rd : RenderingDevice, patterns : String) -> bool:
	if shader.is_valid() and shader_device == rd and shader_patterns == patterns:
		return true
	if shader.is_valid() and shader_device != null:
		shader_device.free_rid(shader)
	shader = RID()
	var source : RDShaderSource = RDShaderSource.new()
	source.source_vertex = VERTEX_SHADER
	source.source_fragment = FRAGMENT_SHADER.replace("@PATTERN_CODE", patterns)
	var spirv : RDShaderSPIRV = rd.shader_compile_spirv_from_source(source)
	if spirv.compile_error_vertex != "" or spirv.compile_error_fragment != "":
		push_error("Splats %s: %s%s" % [ get_hier_name(), spirv.compile_error_vertex, spirv.compile_error_fragment ])
		return false
	shader = rd.shader_create_from_spirv(spirv)
	shader_device = rd
	shader_patterns = patterns
	return shader.is_valid()

func create_pipeline(rd : RenderingDevice, framebuffer : RID, blend_max : bool) -> RID:
	var attachment : RDPipelineColorBlendStateAttachment = RDPipelineColorBlendStateAttachment.new()
	attachment.enable_blend = true
	for f in [ "src_color_blend_factor", "dst_color_blend_factor", "src_alpha_blend_factor", "dst_alpha_blend_factor" ]:
		attachment.set(f, RenderingDevice.BLEND_FACTOR_ONE)
	var op : int = RenderingDevice.BLEND_OP_MAXIMUM if blend_max else RenderingDevice.BLEND_OP_ADD
	attachment.color_blend_op = op
	attachment.alpha_blend_op = op
	var blend : RDPipelineColorBlendState = RDPipelineColorBlendState.new()
	blend.attachments = [ attachment ]
	return rd.render_pipeline_create(shader, rd.framebuffer_get_format(framebuffer), -1,
		RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
		RDPipelineMultisampleState.new(), RDPipelineDepthStencilState.new(), blend)

func in_thread_render(splats : PackedFloat32Array, patterns : String, size : Vector2i) -> bool:
	var rd : RenderingDevice = mm_renderer.rendering_device
	if not compile_shader(rd, patterns):
		return false
	var format : RDTextureFormat = RDTextureFormat.new()
	format.width = size.x
	format.height = size.y
	format.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	var target : RID = rd.texture_create(format, RDTextureView.new())
	var framebuffer : RID = rd.framebuffer_create([ target ])
	var count : int = splats.size()/FLOATS_PER_SPLAT
	var data : PackedByteArray = splats.to_byte_array() if count > 0 else PackedFloat32Array([ 0.0 ]).to_byte_array()
	data.resize(max(data.size(), FLOATS_PER_SPLAT*4))
	var buffer : RID = rd.storage_buffer_create(data.size(), data)
	var uniform : RDUniform = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = 0
	uniform.add_id(buffer)
	var uniform_set : RID = rd.uniform_set_create([ uniform ], shader, 0)
	var pipelines : Array[RID] = [ create_pipeline(rd, framebuffer, false), create_pipeline(rd, framebuffer, true) ]
	var background : float = float(get_parameter("background"))
	# The region to clip space, and its texels: the unit square's (0, 1, 2, -2)
	# and (n, n, 1/n, 1/n)
	var rx : float = float(get_parameter("region_x"))
	var ry : float = float(get_parameter("region_y"))
	var rw : float = float(get_parameter("region_w"))
	var rh : float = float(get_parameter("region_h"))
	var view : PackedFloat32Array = PackedFloat32Array([ rx, ry, 2.0/rw, -2.0/rh, size.x/rw, size.y/rh, rw/size.x, rh/size.y ])
	var draw_list : int = rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_COLOR_ALL, PackedColorArray([ Color(background, background, background, background) ]))
	rd.draw_list_bind_uniform_set(draw_list, uniform_set, 0)
	# Runs of splats of one blending, in order
	var first : int = 0
	while first < count:
		var blend_max : bool = splats[first*FLOATS_PER_SPLAT+13] > 0.5
		var last : int = first+1
		while last < count and (splats[last*FLOATS_PER_SPLAT+13] > 0.5) == blend_max:
			last += 1
		var push : PackedByteArray = view.to_byte_array()
		push.append_array(PackedInt32Array([ first, 0, 0, 0 ]).to_byte_array())
		rd.draw_list_bind_render_pipeline(draw_list, pipelines[1 if blend_max else 0])
		rd.draw_list_bind_uniform_set(draw_list, uniform_set, 0)
		rd.draw_list_set_push_constant(draw_list, push, push.size())
		rd.draw_list_draw(draw_list, false, last-first, 3)
		first = last
	rd.draw_list_end()
	rd.submit()
	rd.sync()
	for rid in pipelines:
		rd.free_rid(rid)
	rd.free_rid(uniform_set)
	rd.free_rid(buffer)
	rd.free_rid(framebuffer)
	texture.set_texture_rid(target, size, format.format, rd)
	return true

func _get_shader_code(uv : String, _output_index : int, context : MMGenContext) -> ShaderCode:
	var genname : String = "o%d" % get_instance_id()
	var rv : ShaderCode = ShaderCode.new()
	rv.output_type = "rgba"
	var variant_index : int = context.get_variant(self, uv)
	if variant_index == -1:
		variant_index = context.get_variant(self, uv)
		rv.add_uniform(get_buffer_name(), "sampler2D", texture)
		rv.code = "vec4 %s_%d = textureLod(%s, fract(%s), 0.0);\n" % [ genname, variant_index, get_buffer_name(), uv ]
	rv.output_values.rgba = "%s_%d" % [ genname, variant_index ]
	return rv

func get_output_attributes(_output_index : int) -> Dictionary:
	return { texture=get_buffer_name() }

func _serialize(data : Dictionary) -> Dictionary:
	data.type = get_type()
	for k in [ "splats", "pattern_code" ]:
		data.parameters[k] = parameters.get(k, "")
	return data
