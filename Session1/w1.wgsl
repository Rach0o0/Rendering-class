struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};
//vertex shader (geometry points)
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
    //gamma: f32,
}; 
@group(0) @binding(0) var<uniform> uniforms : Uniforms;


struct HitInfo{
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    shader: u32,
};

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};

//Light 
struct Light {
    L_i: vec3f, //light intensity
    w_i: vec3f, //light direction
    dist: f32, //distance to light
};

fn get_camera_ray(ipcoords: vec2f) -> Ray
{
    let q = uniforms.b1* ipcoords.x + uniforms.b2*ipcoords.y + uniforms.v*uniforms.cam_const; //we use cam_const here
    let r = Ray(uniforms.eye, normalize(q), 0.001, 1000.0); 
    return r;
}
//fragment shader (pixels and color)
fn intersect_plane(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, position: vec3f, normal:vec3f, color: vec3f) -> bool {
    /*
    plane equation :(p - p0) . n = 0 
    ray equation : r(t) = o + t*w
    t = ((p0 - o).n) / (w.n)
    */
    let denom = dot((*r).direction, normal); // w.n
    if (abs(denom) < 1e-6) {
        return false; //parallel ray (no intersection)
    }

    let t = dot(position - (*r).origin, normal) / denom; //t = ((p0 - o).n) / (w.n)

    if (t > (*r).tmin && t < (*r).tmax && t < (*hit).dist) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = r.origin + t * r.direction;
        (*hit).normal = normalize(normal);
        (*hit).color = color;
        (*hit).shader = 1u; //plane shader

        (*r).tmax = t; //update ray tmax to the intersection point
        return true;
    }
    return false;
}

fn intersect_sphere(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, color: vec3f) -> bool {
    /*
    sphere equation: (p - c).(p - c) - r^2 = 0
    ray equation : r(t) = o + t*w
    quadratic equation: At^2 + Bt + C = 0
    A = w.w
    B = 2w.(o - c)
    C = (o - c).(o - c) - r^2
    t = (-B +- sqrt(B^2 - 4AC)) / 2A
    */
    let oc = (*r).origin - center; //o - c

    let a = dot((*r).direction, (*r).direction); //A = w.w
    let b = 2.0 * dot((*r).direction, oc); //B = 2w.(o - c)
    let c = dot(oc, oc) - radius * radius; //C = (o - c).(o - c) - r^2

    let discriminant = b * b - 4.0 * a * c; //B^2 - 4AC
    if (discriminant < 0.0) {
        return false; //no intersection
    }

    let sqrt_discriminant = sqrt(discriminant);
    var t = (-b - sqrt_discriminant) / (2.0 * a); //t = (-B - sqrt(B^2 - 4AC)) / 2A
    if (t < (*r).tmin || t > (*r).tmax) {
        t = (-b + sqrt_discriminant) / (2.0 * a); //other root
    }

    if (t > (*r).tmin && t < (*r).tmax && t < (*hit).dist) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = (*r).origin + t * (*r).direction;
        (*hit).normal = normalize((*hit).position - center);
        (*hit).color = color; 
        (*hit).shader = 1u; 

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_triangle(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, v0: vec3f, v1: vec3f, v2: vec3f, color: vec3f) -> bool {
    /*
    ray : r(t) = o + t*w
    edges : e0 = v1 - v0
            e1 = v2 - v0
    normal : n = e0 x e1
    t = ((v0 - o).n) / (w.n)
    barycentric coordinates: 
    beta >= 0, gamma >= 0, beta + gamma <= 1
    */
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let n = cross(e0, e1);

    let denom = dot((*r).direction,n); // w.n
    if (abs(denom) < 1e-6) {
        return false; //parallel ray (no intersection)
    }

    let t = dot(v0 - (*r).origin, n) / denom; //t = ((v0 - o).n) / (w.n)
    if (t < (*r).tmin || t > (*r).tmax || t >= (*hit).dist) {
        return false; //no intersection
    }

    let p = (*r).origin + t * (*r).direction; //intersection point
    let v0p = p - v0;

    //barycentric coordinates
    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let d20 = dot(v0p, e0);
    let d21 = dot(v0p, e1);

    let denom_bary = d00 * d11 - d01 * d01;
    let beta = (d11 * d20 - d01 * d21) / denom_bary;
    let gamma = (d00 * d21 - d01 * d20) / denom_bary;
    let alpha = 1.0 - beta - gamma;

    if (beta >= 0.0 && gamma >= 0.0 && alpha >= 0.0) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = p;
        (*hit).normal = normalize(n);
        (*hit).color = color;
        (*hit).shader = 1u; 

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) ->bool {
    //plane
    let plane_pos = vec3f(0.0, 0.0, 0.0);
    let plane_normal = vec3f(0.0, 1.0, 0.0);
    let plane_color = vec3f(0.1, 0.7, 0.0);
    let intersect_plane = intersect_plane(r, hit, plane_pos, plane_normal, plane_color);

    //sphere
    let sphere_center = vec3f(0.0, 0.5, 0.0);
    let sphere_radius = 0.3;
    let sphere_color = vec3f(0.0, 0.0, 0.0);
    let intersect_sphere = intersect_sphere(r, hit, sphere_center, sphere_radius, sphere_color);

    //triangle
    let v0 = vec3f(-0.2, 0.1, 0.9);
    let v1 = vec3f( 0.2, 0.1, 0.9);
    let v2 = vec3f(-0.2, 0.1,-0.1);
    let tri_color = vec3f(0.4, 0.3, 0.2);
    let intersection_triangle = intersect_triangle(r, hit, v0, v1, v2, tri_color);


    return (*hit).has_hit;
}

fn sample_point_light(pos: vec3f) -> Light {
    let light_pos = vec3f(0.0, 1.0, 0.0);
    let intensity = vec3f(3.14159, 3.14159, 3.14159);

    let wi = light_pos - pos;
    let dist = length(wi);
    let dir = wi / dist;

    //Kepler : Li = I / r^2
    let li = intensity / (dist * dist);

    return Light(li, dir, dist);
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_point_light((*hit).position);
    let n_dot_wi = max(dot((*hit).normal, light.w_i), 0.0);
    let lo = ((*hit).color / 3.14159) * light.L_i * n_dot_wi;
    return lo;
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{
    switch hit.shader {
    case 1 { return lambertian(r, hit); }
    case default { return hit.color; }
    }
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{
    let ipcoords = coords*0.5;
    var r = get_camera_ray(ipcoords);

    var hit = HitInfo(false, 1e9, vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u);
    let hit_something = intersect_scene(&r, &hit);

    if (hit_something){
        let shaded_color = shade(&r, &hit);
        return vec4f(shaded_color, 1.0);
    } else {
        return vec4f(0.1, 0.3, 0.6, 1.0);
    }
}