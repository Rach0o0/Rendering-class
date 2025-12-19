//_______________________________________
// STRUCTURES AND UNIFORMS
//_______________________________________

struct Uniforms {
    aspect: f32, //width / height
    cam_const: f32, // zoom / distance image plan
    gamma: f32,
    subdivs: f32,
    eye: vec3f, _pad1: f32, //camera position
    b1: vec3f, _pad2: f32, //horizontal basis camera vector
    b2: vec3f, _pad3: f32, //vertical basis camera vector
    v: vec3f, _pad4: f32,//forward vector
    _pad8: f32, _pad9: f32, width: f32, height: f32,
    frame: f32, progressive: f32, _pad10: f32, _pad11 : f32,
    sunDir: vec3f, _pad5: f32,
    sunRadiance: vec3f, envScale: f32,
    aperture: f32, //radius of lens
    focus_dist: f32,  _pad6: vec2f, //focal distance
    orbitCenter: vec3f, _pad7: f32
}; 

struct Material {
    color: vec3f,
    _pad0: f32,
    emission: vec3f,
    _pad1: f32
};

struct Aabb {
    min: vec3f,
    _pad1: f32,
    max: vec3f,
    _pad2: f32
};

//__________________________________________
// BINDINGS
//__________________________________________

@group(0) @binding(0) var<uniform> uniforms : Uniforms; 
@group(0) @binding(1) var<storage> jitter: array<vec2f>;

@group(0) @binding(2) var<storage> attribs : array<vec4f>;
@group(0) @binding(3) var<storage> meshFaces : array<vec4u>;

@group(0) @binding(4) var<storage> materials : array<Material>;

@group(0) @binding(5) var<storage> bspTree : array<vec4u>;
@group(0) @binding(6) var<storage> treeIds : array<u32>;
@group(0) @binding(7) var<storage> bspPlanes: array<f32>;
@group(0) @binding(8) var<uniform> aabb: Aabb;
@group(0) @binding(9) var<storage> lightIndices : array<u32>;
@group(0) @binding(11) var envMap : texture_2d<u32>;
@group(0) @binding(12) var envSampler : sampler;

//_______________________________________________
//CONSTANTS FOR BSP
//_______________________________________________
const BSP_X_AXIS : u32 = 0u;
const BSP_Y_AXIS : u32 = 1u;
const BSP_Z_AXIS : u32 = 2u;
const BSP_LEAF   : u32 = 3u;
const MAX_LEVEL  : u32 = 20u;


//__________________________________________
// VERTEX SHADER
//__________________________________________
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



//__________________________________________
// RAY STRUCTURES
//__________________________________________
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};


struct HitInfo{
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,

    emission: vec3f, 
    diffuse: vec3f,
    specular: vec3f,

    continue_ray: bool, 
    ior : f32, 
    texcoords: vec2f, 

    object_id: u32,
    emit_flag : u32,
    throughput : vec3f,  
    extinction: vec3f,
};

fn reset_hit(hit: ptr<function, HitInfo>, tmax: f32 ){
    (*hit).has_hit = false;
    (*hit).dist = tmax;
    (*hit).position = vec3f(0.0);
    (*hit).normal = vec3f(0.0);

    (*hit).emission = vec3f(0.0);
    (*hit).diffuse = vec3f(0.0);
    (*hit).specular = vec3f(0.0);

    (*hit).continue_ray = false;
    (*hit).ior = 1.0;
    (*hit).texcoords = vec2f(0.0);
    (*hit).object_id = 0u;
    (*hit).emit_flag = 1u;
    (*hit).throughput = vec3f(1.0);
    (*hit).extinction = vec3f(0.0);
}

//_______________________________________
// LIGHT STRUCTURE AND CONSTANTS
//_______________________________________

//light 
struct Light {
    L_i: vec3f, 
    w_i: vec3f,
    dist: f32, 
};


//bulb placement (relative to orbitCenter)
const BULB_POS      = vec3f(2.0, 1.5, 2.0);

const BULB_R_OUTER  : f32 = 0.35;  //glass shell radius
const BULB_R_INNER  : f32 = 0.18;  //emissive core radius (smaller)

//warm orange emission
const BULB_EMISSION = vec3f(60.0, 35.0, 12.0); // warm/orange

const NUM_BULBS : u32 = 4u;

const BULB_OFFSETS = array<vec3f, NUM_BULBS>(
    vec3f( 2.0, 1.5,  2.0),
    vec3f(-2.0, 1.2,  1.5),
    vec3f( 1.5, 1.8, -2.0),
    vec3f(-1.8, 1.4, -1.8)
);

const GLOSSY_WEIGHT = 0.15;
const GLOSSY_ROUGHNESS = 0.18;



//__________________________________________
// GENERATE CAMERA RAY
//__________________________________________
fn get_camera_ray(ipcoords: vec2f) -> Ray
{
    let q = uniforms.b1* ipcoords.x + uniforms.b2*ipcoords.y + uniforms.v*uniforms.cam_const; //we use cam_const here
    let r = Ray(uniforms.eye, normalize(q), 0.001, 10000.0); 
    return r;
}


fn get_camera_ray_dof(ipcoords: vec2f, t: ptr<function, u32>) -> Ray {
    let dir = normalize(
        uniforms.b1 * ipcoords.x +
        uniforms.b2 * ipcoords.y +
        uniforms.v  * uniforms.cam_const
    );

    let focus_point = uniforms.eye + dir * uniforms.focus_dist;

    let r = sqrt(rnd(t));
    let theta = 2.0 * 3.14159265 * rnd(t);
    let lens_offset =
        uniforms.b1 * (r * cos(theta) * uniforms.aperture * 3.0) +
        uniforms.b2 * (r * sin(theta) * uniforms.aperture * 3.0);

    let origin = uniforms.eye + lens_offset;
    let new_dir = normalize(focus_point - origin);

    return Ray(origin, new_dir, 0.001, 1e9);
}


//__________________________
// MAPPING PANORAMIC
//_______________________
fn direction_to_panorama_uv(d: vec3f) -> vec2f {
    let dir = normalize(d);

    let u = 0.5 + atan2(dir.z, dir.x) / (2.0 * 3.14159265);
    let v = 0.5 - asin(dir.y) / 3.14159265;

    return vec2f(u, v);
}

//__________________________
// HELPER FOR READING ENVMAP
//__________________________

fn decode_rgbE(rgba: vec4u) -> vec3f {
    if (rgba.w == 0u) {
        return vec3f(0.0);
    }
    let e = f32(rgba.w) - 128.0;
    let f = exp2(e - 8.0);
    return vec3f(
        f32(rgba.x),
        f32(rgba.y),
        f32(rgba.z)
    ) * f;
}

fn sample_env(dir: vec3f) -> vec3f {
    let uv = direction_to_panorama_uv(dir);

    let dims = textureDimensions(envMap);             
    let size_f = vec2f(f32(dims.x), f32(dims.y));

    var st = uv * size_f;
    st = clamp(st, vec2f(0.0), size_f - vec2f(1.0));  

    let ij = vec2u(st);                               
    let texel : vec4u = textureLoad(envMap, ij, 0); 

    return decode_rgbE(texel) * uniforms.envScale;
}




//________________________
// FRESNEL R
//______________________

fn fresnel_R(cos_i : f32, cos_t : f32, eta : f32) -> f32 {
    // eta = ni / nt
    if (cos_t < 0.0) {
        return 1.0;
    }

    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let r_par = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);

    return 0.5 * (r_perp * r_perp + r_par * r_par);
}



//________________________________________
// INTERSECTIONS
//________________________________________

fn intersect_sphere(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, emission: vec3f, diffuse: vec3f) -> bool {
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
        (*hit).emission = emission;
        (*hit).diffuse = diffuse;

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_triangle(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, face_index:u32, emission: vec3f, diffuse: vec3f) -> bool {
    //take 3 indexed vertices
    let face = meshFaces[face_index];
    let v0i = face.x;
    let v1i = face.y;
    let v2i = face.z;

    //vertex positions
    let v0 = attribs[v0i * 2u + 0u].xyz;
    let v1 = attribs[v1i * 2u + 0u].xyz;
    let v2 = attribs[v2i * 2u + 0u].xyz;

    //vertex normals
    let n0 = attribs[v0i * 2u + 1u].xyz;
    let n1 = attribs[v1i * 2u + 1u].xyz;
    let n2 = attribs[v2i * 2u + 1u].xyz;

    //ray-triangle intersection
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
        (*hit).normal = normalize(alpha * n0 + beta * n1 + gamma * n2);
        (*hit).emission = emission;
        (*hit).diffuse = diffuse;

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_aabb(r: ptr<function, Ray>) -> bool {
    let p1 = (aabb.min - (*r).origin) / (*r).direction;
    let p2 = (aabb.max - (*r).origin) / (*r).direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let box_tmin = max(pmin.x, max(pmin.y, pmin.z)) - 1.0e-3f;
    let box_tmax = min(pmax.x, min(pmax.y, pmax.z)) + 1.0e-3f;
    if (box_tmin > box_tmax || box_tmin > (*r).tmax || box_tmax < (*r).tmin) {
        return false;
    }
    (*r).tmin = max(box_tmin, (*r).tmin);
    (*r).tmax = min(box_tmax, (*r).tmax);
    return true;
}

fn intersect_bsp(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var branch_node: array<vec2u, MAX_LEVEL>;
    var branch_ray: array<vec2f, MAX_LEVEL>;
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;

    for (var i = 0u; i <= MAX_LEVEL; i++) {
        let tree_node = bspTree[node];
        let node_axis_leaf = tree_node.x & 3u;

        if (node_axis_leaf == BSP_LEAF) {
            let node_count = tree_node.x >> 2u;
            let node_id = tree_node.y;
            var found = false;
            for (var j = 0u; j < node_count; j++) {
                let obj_idx = treeIds[node_id + j];
                let matIdx = meshFaces[obj_idx].w;
                let mat = materials[matIdx];
                if (intersect_triangle(r, hit, obj_idx, mat.emission, mat.color)) {
                    (*r).tmax = (*hit).dist;
                    found = true;
                }
            }
            if (found) { return true; }
            else if (branch_lvl == 0u) { return false; }
            else {
                branch_lvl--;
                i = branch_node[branch_lvl].x;
                node = branch_node[branch_lvl].y;
                (*r).tmin = branch_ray[branch_lvl].x;
                (*r).tmax = branch_ray[branch_lvl].y;
                continue;
            }
        }

        let axis_direction = (*r).direction[node_axis_leaf];
        let axis_origin = (*r).origin[node_axis_leaf];
        if (axis_direction >= 0.0f) {
            near_node = tree_node.z;
            far_node = tree_node.w;
        } else {
            near_node = tree_node.w;
            far_node = tree_node.z;
        }

        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin) / denom;

        if (t > (*r).tmax) { node = near_node; }
        else if (t < (*r).tmin) { node = far_node; }
        else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = (*r).tmax;
            branch_lvl++;
            (*r).tmax = t;
            node = near_node;
        }
    }
    return false;
}

fn intersect_plane_y0(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    let denom = (*r).direction.y;
    if (abs(denom) < 1e-6) { return false; }

    let t = -(*r).origin.y / denom;
    if (t < (*r).tmin || t > (*r).tmax) { return false; }

    (*hit).has_hit = true;
    (*hit).dist = t;
    (*hit).position = (*r).origin + t * (*r).direction;
    (*hit).normal = vec3f(0.0, 1.0, 0.0);
    (*hit).emission = vec3f(0.0);
    (*hit).diffuse = vec3f(0.5, 0.5, 0.5); // gris

    return true;
}



fn intersect_scene(
    r: ptr<function, Ray>,
    hit: ptr<function, HitInfo>,
    isPrimary: bool
) -> bool {

    let r0 = *r;
    var found = false;

    var best: HitInfo;
    reset_hit(&best, r0.tmax);

    //___________________________
    // LAMPS (core + glass)
    //___________________________
    for (var i = 0u; i < NUM_BULBS; i++) {

        let center = uniforms.orbitCenter + BULB_OFFSETS[i];

        //emissive core
        var hit_core: HitInfo;
        reset_hit(&hit_core, r0.tmax);
        var r_core = r0;

        if (intersect_sphere(
            &r_core,
            &hit_core,
            center,
            BULB_R_INNER,
            BULB_EMISSION,
            vec3f(0.0)
        )) {
            if (hit_core.dist < best.dist) {
                hit_core.object_id = 100u + i; // <<<<< IMPORTANT
                hit_core.emit_flag = 1u;
                best = hit_core;
                found = true;
            }
        }

        //glass shell
        var hit_glass: HitInfo;
        reset_hit(&hit_glass, r0.tmax);
        var r_glass = r0;

        if (intersect_sphere(
            &r_glass,
            &hit_glass,
            center,
            BULB_R_OUTER,
            vec3f(0.0),
            vec3f(0.0)
        )) {
            if (hit_glass.dist < best.dist) {
                hit_glass.object_id = 200u + i; // <<<<< IMPORTANT
                hit_glass.ior = 1.5;
                hit_glass.extinction = vec3f(0.02, 0.04, 0.08);
                hit_glass.emit_flag = 0u;
                best = hit_glass;
                found = true;
            }
        }
    }

    //____________________________________
    // MESH (BSP)
    //____________________________________
    var hit_tri: HitInfo;
    var got_tri = false;

    if (isPrimary) {
        var r_box = r0;
        if (intersect_aabb(&r_box)) {
            reset_hit(&hit_tri, r_box.tmax);
            var r_tri = r_box;
            got_tri = intersect_bsp(&r_tri, &hit_tri);
        }
    } else {
        reset_hit(&hit_tri, r0.tmax);
        var r_tri2 = r0;
        got_tri = intersect_bsp(&r_tri2, &hit_tri);
    }

    if (got_tri && hit_tri.dist < best.dist) {
        hit_tri.object_id = 1u;
        best = hit_tri;
        found = true;
    }

    if (found) {
        *hit = best;
        (*r).tmax = best.dist;
        return true;
    }

    return false;
}



//______________________
// CAUSTIC
//______________________

fn fake_caustic(hitPos: vec3f, hitN: vec3f) -> vec3f {
    var C = vec3f(0.0);

    for (var i = 0u; i < NUM_BULBS; i++) {
        let center = uniforms.orbitCenter + BULB_OFFSETS[i];
        let wi = center - hitPos;
        let dist2 = dot(wi, wi);
        let dir = normalize(wi);

        let ndl = max(dot(hitN, dir), 0.0);
        let focus = pow(ndl, 5.0);

        C += BULB_EMISSION * focus / dist2;
    }
    return C;
}




//_________________________________________
// LIGHT SAMPLING
//_________________________________________


fn sample_directional_light() -> Light {
    let dir = normalize(uniforms.sunDir);
    let L = uniforms.sunRadiance;
    return Light(L, dir, 1e9);
}






//__________________________________________
// SHADING FUNCTIONS
//__________________________________________

fn lambertian(
    r: ptr<function, Ray>,
    hit: ptr<function, HitInfo>,
    t: ptr<function, u32>
) -> vec3f {

    var Lo = vec3f(0.0);
    let n = normalize(hit.normal);

    //__________________________
    // SUN (directional)
    //__________________________
    let sun = sample_directional_light();
    let l = normalize(sun.w_i);
    let ndotl = max(dot(n, l), 0.0);

    if (ndotl > 0.0) {
        var sRay = Ray(hit.position + 0.001 * n, l, 0.001, 1e9);
        var sHit: HitInfo;
        reset_hit(&sHit, 1e9);

        if (!intersect_scene(&sRay, &sHit, false)) {
            Lo += (hit.diffuse / 3.14159) * sun.L_i * ndotl;
        }
    }

    //_____________________________
    // 2) BULBS (spherical lights)
    //___________________________
    for (var i = 0u; i < NUM_BULBS; i++) {

        let center = uniforms.orbitCenter + BULB_OFFSETS[i];
        let wi = center - hit.position;
        let dist2 = dot(wi, wi);
        let dist = sqrt(dist2);
        let dir = wi / dist;

        let ndl = max(dot(n, dir), 0.0);
        if (ndl <= 0.0) { continue; }

        // shadow ray
        var sRay = Ray(hit.position + 0.001 * n, dir, 0.001, dist - 0.002);
        var sHit: HitInfo;
        reset_hit(&sHit, 1e9);

        if (intersect_scene(&sRay, &sHit, false)) {
            let isLamp =
                (sHit.object_id >= 100u && sHit.object_id < 104u) ||
                (sHit.object_id >= 200u && sHit.object_id < 204u);

            if (!isLamp) {
                continue;
            }
        }

        let Li = BULB_EMISSION / dist2;
        Lo += (hit.diffuse / 3.14159) * Li * ndl;
    }

    //_________________________
    // DIRECT EMISSION
    //_________________________
    if (hit.emit_flag == 1u) {
        Lo += hit.emission;
    }

    //_____________________________
    // FAKE CAUSTIC 
    //____________________________
    let caustic = fake_caustic(hit.position, n);
    Lo += caustic * 0.25;

    hit.continue_ray = false;
    return Lo;
}


fn transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function,u32>) -> vec3f {

    let n = normalize(hit.normal);
    let d = normalize((*r).direction);

    var ni = 1.0;
    var nt = hit.ior;
    var normal = n;
    var cos_i = dot(-d, n);

    //detection input/output
    var entering = true;

    if (cos_i < 0.0) {
        entering = false;   
        cos_i = -cos_i;
        normal = -n;
        ni = hit.ior;
        nt = 1.0;
    }

    let eta = ni / nt;
    let k = 1.0 - eta * eta * (1.0 - cos_i*cos_i);

    var cos_t = -1.0;
    if (k >= 0.0) {
        cos_t = sqrt(k);
    }

    //fresnel
    let R = fresnel_R(cos_i, cos_t, eta);

    //reflection ?
    if (R > 0.5 || k < 0.0) {
        let refl = normalize(d - 2.0 * dot(d, normal) * normal);
        (*r).origin = hit.position + refl * 0.001;
        (*r).direction = refl;
        (*r).tmin = 0.001;
        (*r).tmax = 1e9;
        hit.continue_ray = true;
        return vec3f(0.0);
    }

    // tRANSMISSION
    let refr = normalize(eta * d + (eta*cos_i - cos_t)*normal);
    (*r).origin = hit.position + refr * 0.001;
    (*r).direction = refr;
    (*r).tmin = 0.001;
    (*r).tmax = 1e9;

    //absorption
    if (!entering) {
        let s = 0.2 * 90.0;
        let Tr = exp(-hit.extinction * s);
        let p = (Tr.x + Tr.y + Tr.z) / 3.0;

        if (rnd(t) > p) {
            hit.continue_ray = false;
            return vec3f(0.0);
        }

        hit.throughput *= Tr / p;
    }

    if (entering) {
        hit.throughput *= exp(-hit.extinction * 0.05 * 90.0); // petite absorption
    }


    hit.continue_ray = true;
    return vec3f(0.0);
}


fn shade(
    r: ptr<function, Ray>,
    hit: ptr<function, HitInfo>,
    t: ptr<function, u32>
) -> vec3f {

    //emissive core
    if (hit.object_id >= 100u && hit.object_id < 104u) {

        let idx = hit.object_id - 100u;
        let center = uniforms.orbitCenter + BULB_OFFSETS[idx];

        let p = hit.position - center;
        let d = length(p);
        let x = clamp(d / BULB_R_INNER, 0.0, 1.0);

        let t = f32(uniforms.frame) * 0.01;

        let time =
            sin(t * 1.0) +
            0.7 * sin(t * 1.618) +   //gold number
            0.4 * sin(t * 2.414) +
            0.2 * sin(t * 3.14159);


        //noise 
        let noise =
            sin(12.0 * p.x + 3.0 * p.y + time) *
            sin(9.0  * p.y + 4.0 * p.z - time * 1.2) *
            sin(15.0 * p.z + 2.0 * p.x + time * 0.7);

        //radial profils
        let core = exp(-10.0 * x * x);
        let filaments = abs(noise) * exp(-4.0 * x);
        let halo = exp(-3.0 * x);

        //final combination
        let intensity =
            2.5 * core
            + 1.8 * filaments  
            + 0.35 * halo;    

        hit.continue_ray = false;
        return BULB_EMISSION * intensity;
    }


    //____________
    // GLASS
    // ___________
    if (hit.object_id >= 200u && hit.object_id < 204u) {

        transparent(r, hit, t);

        let view = normalize(-r.direction);
        let n = normalize(hit.normal);

        let fres = pow(1.0 - max(dot(view, n), 0.0), 3.0);

        let glassTint = vec3f(0.6, 0.75, 1.0); // bleu-gris clair

        let absorption = glassTint * 0.08;

        let edge = fres * glassTint * 0.25;

        hit.continue_ray = true;

        return absorption + edge;
    }
    

    //rest of the scene
    return lambertian(r, hit, t);
}



//______________________________________________
// RANDOM NUMBER GENERATOR
//_________________________________________
fn tea(val0: u32, val1: u32) -> u32 {
    const N = 16u;
    var v0 = val0;
    var v1 = val1;
    var s0 = 0u;
    for (var n = 0u; n < N; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

fn mcg31(prev: ptr<function, u32>) -> u32 {
    const LCG_A = 1977654935u;
    *prev = (LCG_A * (*prev)) & 0x7FFFFFFFu;
    return *prev;
}

fn rnd(prev: ptr<function, u32>) -> f32 {
    return f32(mcg31(prev)) / f32(0x80000000u);
}

//___________________________________________
// COSINE-WEIGHTED HEMISPHERE SAMPLING
//____________________________________________
fn sample_cosine_weighted(t : ptr<function, u32>) -> vec3f {
    let u1 = rnd(t);
    let u2 = rnd(t);

    let r = sqrt(u1);
    let theta = 2.0 * 3.14159265 * u2;

    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(1.0 - u1);

    return vec3f(x, y, z); 
}

fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
    let s = sign(n.z + 1e-6);
    let a = -1.0 / (1.0 + abs(n.z));
    let b = n.x * n.y * a;

    return vec3f(1.0 + n.x*n.x*a, b, -s*n.x) * v.x + vec3f(s*b, s*(1.0 + n.y*n.y*a), -n.y) * v.y + n * v.z;
}


//__________________________________________
// FRAGMENT SHADER
//__________________________________________

@fragment
fn main_fs(
    @builtin(position) fragcoord: vec4f,
    @location(0) coords: vec2f
) -> @location(0) vec4f
{
    let ipcoords = coords * 0.5;

    // random seed
    let launch_idx =
        u32(fragcoord.y) * u32(uniforms.width) +
        u32(fragcoord.x);
    var t = tea(launch_idx, u32(uniforms.frame));

    var final_color = vec3f(0.0);

    let n = i32(uniforms.subdivs * uniforms.subdivs);
    let gamma = uniforms.gamma;

    for (var k = 0; k < n; k++) {

        // jitter
        let offset = jitter[k];
        let jittered = ipcoords + offset;

        var r = get_camera_ray_dof(jittered, &t);

        var color = vec3f(0.0);
        var throughput = vec3f(1.0);

        var saw_emitter = false;

        for (var depth = 0; depth < 5; depth++) {

            var hit: HitInfo;
            reset_hit(&hit, 1e9);

            let hit_something = intersect_scene(&r, &hit, depth == 0);

            if (depth > 0) {
                hit.emit_flag = 0u;
            }

            if (!hit_something) {
                color += throughput * sample_env(r.direction);
                break;
            }

            if (hit.object_id >= 100u && hit.object_id < 104u) {
                saw_emitter = true;
            }

            hit.throughput = throughput;
            let shaded = shade(&r, &hit, &t);
            color += hit.throughput * shaded;

            if (hit.object_id >= 200u && hit.object_id < 204u) {
                throughput = hit.throughput;          // ne pas multiplier par diffuse
            } else {
                throughput = hit.throughput * hit.diffuse; // mesh diffuse
            }

            if (!hit.continue_ray) {
                break;
            }
        }

        final_color += color;

        //halo
        if (!saw_emitter) {
            for (var i = 0u; i < NUM_BULBS; i++) {

                let center = uniforms.orbitCenter + BULB_OFFSETS[i];
                let v = center - r.origin;
                let t_proj = dot(v, r.direction);

                if (t_proj > 0.0) {
                    let closest = r.origin + t_proj * r.direction;
                    let d = length(closest - center);

                    let halo = exp(-8.0 * d * d);
                    final_color += BULB_EMISSION * halo * 0.01;
                }
            }
        }
    }

    final_color /= f32(n);
    final_color = pow(final_color, vec3f(1.0 / gamma));

    return vec4f(final_color, 1.0);
}

