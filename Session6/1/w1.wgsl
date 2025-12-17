//_______________________________________
// STRUCTURES AND UNIFORMS
//_______________________________________

struct Uniforms {
    aspect: f32, // width / height
    cam_const: f32, // zoom / distance image plane
    gamma: f32,
    subdivs: f32,
    eye: vec3f, _pad1: f32, // camera position
    b1: vec3f, _pad2: f32, // horizontal basis camera vector
    b2: vec3f, _pad3: f32, // vertical basis camera vector
    v: vec3f, _pad4: f32, // forward vector
    sphereShader: f32,
    mattShader: f32,
    _pad5: vec2f,
};

struct Material {
    color: vec3f, // diffuse reflectance (rho_d)
    _pad0: f32,
    emission: vec3f, // emitted radiance (Le)
    _pad1: f32,
};

struct Aabb {
    min: vec3f,
    _pad1: f32,
    max: vec3f,
    _pad2: f32,
};

//__________________________________________
// BINDINGS
//__________________________________________

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> jitter: array<vec2f>;
@group(0) @binding(2) var<storage> vPositions: array<vec3f>;
@group(0) @binding(3) var<storage> meshFaces: array<vec3u>;
@group(0) @binding(4) var<storage> vNormals: array<vec3f>;
@group(0) @binding(5) var<storage> matIndices: array<u32>;
@group(0) @binding(6) var<storage> materials: array<Material>;
@group(0) @binding(7) var<storage> lightIndices: array<u32>;
@group(0) @binding(10) var<uniform> aabb: Aabb;

//__________________________________________
// VERTEX SHADER
//__________________________________________

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,
};

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
    const pos = array<vec2f, 4>(
        vec2f(-1.0, 1.0),
        vec2f(-1.0, -1.0),
        vec2f(1.0, 1.0),
        vec2f(1.0, -1.0)
    );

    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

//__________________________________________
// RAY STRUCTURES
//__________________________________________

struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};

struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    emission: vec3f,
    diffuse: vec3f,
    specular: vec3f,
    shader: u32,
    continue_ray: bool,
    ior: f32,
    texcoords: vec2f,
    object_id: u32,
};

fn reset_hit(hit: ptr<function, HitInfo>, tmax: f32) {
    (*hit).has_hit = false;
    (*hit).dist = tmax;
    (*hit).position = vec3f(0.0);
    (*hit).normal = vec3f(0.0);
    (*hit).emission = vec3f(0.0);
    (*hit).diffuse = vec3f(0.0);
    (*hit).specular = vec3f(0.0);
    (*hit).shader = 0u;
    (*hit).continue_ray = false;
    (*hit).ior = 1.0;
    (*hit).texcoords = vec2f(0.0);
}

//__________________________________________
// ONB STRUCTURE
//__________________________________________

struct Onb {
    tangent: vec3f,
    binormal: vec3f,
    normal: vec3f,
};

const plane_onb = Onb(
    vec3f(-1.0, 0.0, 0.0),
    vec3f(0.0, 0.0, 1.0),
    vec3f(0.0, 1.0, 0.0)
);

//_______________________________________
// LIGHT STRUCTURE AND CONSTANTS
//_______________________________________

struct Light {
    L_i: vec3f,
    w_i: vec3f,
    dist: f32,
};

const SHADER_LAMBERTIAN: u32 = 1u;
const SHADER_PHONG: u32 = 2u;
const SHADER_MIRROR: u32 = 3u;
const SHADER_REFRACT: u32 = 4u;
const SHADER_GLOSSY: u32 = 5u;

//__________________________________________
// CAMERA RAY
//__________________________________________

fn get_camera_ray(ipcoords: vec2f) -> Ray {
    let q = uniforms.b1 * ipcoords.x + uniforms.b2 * ipcoords.y + uniforms.v * uniforms.cam_const;
    return Ray(uniforms.eye, normalize(q), 0.001, 10000.0);
}

//________________________________________
// INTERSECTIONS
//________________________________________

fn intersect_plane(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb, emission: vec3f, diffuse: vec3f, specular: vec3f, shade: u32) -> bool {
    let denom = dot((*r).direction, onb.normal);
    if (abs(denom) < 1e-6) { return false; }

    let t = dot(position - (*r).origin, onb.normal) / denom;
    if (t > (*r).tmin && t < (*r).tmax && t < (*hit).dist) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = (*r).origin + t * (*r).direction;
        (*hit).normal = normalize(onb.normal);
        (*hit).emission = emission;
        (*hit).diffuse = diffuse;
        (*hit).specular = specular;
        (*hit).shader = shade;
        (*r).tmax = t;
        return true;
    }
    return false;
}

fn intersect_triangle(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, face_index: u32, emission: vec3f, diffuse: vec3f, specular: vec3f, shade: u32) -> bool {
    let face = meshFaces[face_index];
    let v0 = vPositions[face.x];
    let v1 = vPositions[face.y];
    let v2 = vPositions[face.z];

    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let n = cross(e0, e1);

    let denom = dot((*r).direction, n);
    if (abs(denom) < 1e-6) { return false; }

    let t = dot(v0 - (*r).origin, n) / denom;
    if (t < (*r).tmin || t > (*r).tmax || t >= (*hit).dist) { return false; }

    let p = (*r).origin + t * (*r).direction;
    let v0p = p - v0;

    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let d20 = dot(v0p, e0);
    let d21 = dot(v0p, e1);

    let n0 = vNormals[face.x];
    let n1 = vNormals[face.y];
    let n2 = vNormals[face.z];

    let denom_bary = d00 * d11 - d01 * d01;
    let beta = (d11 * d20 - d01 * d21) / denom_bary;
    let gamma = (d00 * d21 - d01 * d20) / denom_bary;
    let alpha = 1.0 - beta - gamma;

    if (beta >= 0.0 && gamma >= 0.0 && alpha >= 0.0) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = p;
        (*hit).normal = normalize(alpha * n0 + beta * n1 + gamma * n2);
        (*hit).emission = emission;
        (*hit).diffuse = diffuse;
        (*hit).specular = specular;
        (*hit).shader = shade;
        (*r).tmax = t;
        return true;
    }
    return false;
}

fn intersect_aabb(r: Ray, box: Aabb) -> bool {
    var tmin = (box.min.x - r.origin.x) / r.direction.x;
    var tmax = (box.max.x - r.origin.x) / r.direction.x;
    if (tmin > tmax) { let tmp = tmin; tmin = tmax; tmax = tmp; }

    var tymin = (box.min.y - r.origin.y) / r.direction.y;
    var tymax = (box.max.y - r.origin.y) / r.direction.y;
    if (tymin > tymax) { let tmp = tymin; tymin = tymax; tymax = tmp; }

    if ((tmin > tymax) || (tymin > tmax)) { return false; }
    if (tymin > tmin) { tmin = tymin; }
    if (tymax < tmax) { tmax = tymax; }

    var tzmin = (box.min.z - r.origin.z) / r.direction.z;
    var tzmax = (box.max.z - r.origin.z) / r.direction.z;
    if (tzmin > tzmax) { let tmp = tzmin; tzmin = tzmax; tzmax = tmp; }

    if ((tmin > tzmax) || (tzmin > tmax)) { return false; }
    return true;
}

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    if (!intersect_aabb(*r, aabb)) { return false; }

    for (var i: u32 = 0u; i < arrayLength(&meshFaces); i = i + 1u) {
        let matIdx = matIndices[i];
        let mat = materials[matIdx];
        if (intersect_triangle(r, hit, i, mat.emission, mat.color, vec3f(0.0), SHADER_LAMBERTIAN)) {
            (*hit).object_id = 1u;
        }
    }
    return (*hit).has_hit;
}

//__________________________________________
// LIGHT SAMPLING
//__________________________________________

fn sample_point_light(pos: vec3f) -> Light {
    let light_pos = vec3f(0.0, 1.0, 0.0);
    let intensity = vec3f(3.14159);
    let wi = light_pos - pos;
    let dist = length(wi);
    let dir = wi / dist;
    let li = intensity / (dist * dist);
    return Light(li, dir, dist);
}

fn sample_area_light(hitPos: vec3f, hitN: vec3f) -> Light {
    var L_accum = vec3f(0.0);
    var dir_accum = vec3f(0.0);
    var total_weight = 0.0;

    for (var i: u32 = 0u; i < arrayLength(&lightIndices); i = i + 1u) {
        let triIdx = lightIndices[i];
        let face = meshFaces[triIdx];
        let v0 = vPositions[face.x];
        let v1 = vPositions[face.y];
        let v2 = vPositions[face.z];

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let nL = normalize(cross(e1, e2));
        let area = 0.5 * length(cross(e1, e2));
        let center = (v0 + v1 + v2) / 3.0;

        let wi = center - hitPos;
        let dist2 = dot(wi, wi);
        let dist = sqrt(dist2);
        let dir = wi / dist;

        let cos_i = max(dot(hitN, dir), 0.0);
        let cos_o = max(dot(-dir, nL), 0.0);

        var shadowRay = Ray(hitPos, dir, 0.001, dist - 0.001);
        var shadowHit: HitInfo;
        reset_hit(&shadowHit, 1e9);
        if (intersect_scene(&shadowRay, &shadowHit)) { continue; }

        let matIdx = matIndices[triIdx];
        let mat = materials[matIdx];
        let Le = mat.emission;

        let dE = Le * (cos_i * cos_o / (3.14159 * dist2)) * area;

        L_accum += dE;
        dir_accum += dir * cos_i;
        total_weight += cos_i;
    }

    var avg_dir = vec3f(0.0);
    if (total_weight > 0.0) {
        avg_dir = normalize(dir_accum / total_weight);
    }
    return Light(L_accum, avg_dir, 1.0);
}

//__________________________________________
// SHADING FUNCTIONS
//__________________________________________

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_area_light((*hit).position, (*hit).normal);
    let n_dot_wi = max(dot((*hit).normal, light.w_i), 0.0);
    let kd = (*hit).diffuse;
    let Le = (*hit).emission;
    return Le + (kd / 3.14159) * light.L_i * n_dot_wi;
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    return lambertian(r, hit);
}

//__________________________________________
// FRAGMENT SHADER
//__________________________________________

@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    let ipcoords = coords * 0.5;
    var final_color = vec3f(0.0);
    let n = i32(uniforms.subdivs * uniforms.subdivs);
    let gamma = uniforms.gamma;

    for (var k = 0; k < n; k++) {
        let offset = jitter[k];
        let jittered = ipcoords + offset;
        var r = get_camera_ray(jittered);
        var hit: HitInfo;
        var color = vec3f(0.0);

        for (var depth = 0; depth < 5; depth++) {
            reset_hit(&hit, 1e9);
            let hit_something = intersect_scene(&r, &hit);
            if (!hit_something) {
                color += vec3f(0.1, 0.1, 0.1);
                break;
            }
            color += shade(&r, &hit);
            if (!hit.continue_ray) { break; }
        }
        final_color += color;
    }

    final_color /= f32(n);
    let corrected = pow(final_color, vec3f(1.0 / gamma));
    return vec4f(corrected, 1.0);
}
