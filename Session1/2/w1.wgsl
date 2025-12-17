struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};
@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

struct Uniforms {
aspect: f32,
cam_const: f32,
eye: vec3f,
b1: vec3f,
b2: vec3f,
v: vec3f,
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;


// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};
fn get_camera_ray(ipcoords: vec2f) -> Ray
{
    let q = uniforms.b1* ipcoords.x + uniforms.b2*ipcoords.y + uniforms.v*uniforms.cam_const; //we use cam_const here
    let r = Ray(uniforms.eye, normalize(q), 0.001, 1000.0); 
    return r;
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{
    let ipcoords = coords*0.5;
    var r = get_camera_ray(ipcoords);
    return vec4f(r.direction*0.5 + 0.5, 1.0);
}