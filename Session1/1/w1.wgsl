struct VSOut {
    @builtin(position) position: vec4f, 
    @location(0) coords : vec2f, 
};

//vertex shader
@vertex
//GPU calls this function for each vertex to draw
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex]; //between -1 and 1
    return vsOut;
}

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};
fn get_camera_ray(ipcoords: vec2f) -> Ray
{
    // Implement ray generation (WGSL has vector operations like normalize and cross)
    //camera parameters
    let e = vec3f(2.0, 1.5, 2.0); //eye point
    let p = vec3f(0.0, 0.5, 0.0); //view point
    let u = vec3f(0.0, 1.0, 0.0); //up direction
    let d = 1.0; //distance to image plane

    //camera basis
    let v = normalize(p - e); //view direction
    let b1 = normalize(cross(v, u)); //right direction
    let b2 = cross(b1, v); //real up direction

    //image plane coordinates
    let x = ipcoords.x; //[-1,1]
    let y = ipcoords.y; //[-1,1]

    //ray
    let q = b1*x + b2*y + v*d; //point on image plane
    let r = Ray(e, normalize(q), 0.001, 1000.0); //ray
    return r;
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{
    let ipcoords = coords*0.5; //stay in the window
    var r = get_camera_ray(ipcoords); //generate the ray
    return vec4f(r.direction*0.5 + 0.5, 1.0);
}