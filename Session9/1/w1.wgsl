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
    sphereShader: f32, mattShader: f32, width: f32, height: f32,
    frame: f32, progressive: f32, bgFlag: f32, objectShader : f32
}; 

struct Material {
    color: vec3f,   // diffuse reflectance (rho_d)
    _pad0: f32,
    emission: vec3f, // emitted radiance (Le)
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

@group(0) @binding(0) var<uniform> uniforms : Uniforms; //all info on uniforms from js
@group(0) @binding(1) var<storage> jitter: array<vec2f>;
//interleaved position + normal per vertex
@group(0) @binding(2) var<storage> attribs : array<vec4f>;
//meshFaces : each face = vec4y(x,y,z, matIndex)
@group(0) @binding(3) var<storage> meshFaces : array<vec4u>;
//materials buffer
@group(0) @binding(4) var<storage> materials : array<Material>;
//BSP tree data
@group(0) @binding(5) var<storage> bspTree : array<vec4u>;
@group(0) @binding(6) var<storage> treeIds : array<u32>;
@group(0) @binding(7) var<storage> bspPlanes: array<f32>;
@group(0) @binding(8) var<uniform> aabb: Aabb;
@group(0) @binding(9) var<storage> lightIndices : array<u32>;
@group(0) @binding(11) var envMap : texture_2d<f32>;
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
// CONSTANTS SPHERES
//___________________________________________
const EPS : f32 = 0.001;

//center, radius
const SPH_L_C : vec3f = vec3f(420.0, 90.0, 370);
const SPH_L_R : f32 = 90.0;

const SPH_R_C : vec3f = vec3f(130.0, 90.0, 250.0);
const SPH_R_R : f32 = 90.0;
//__________________________________________
// VERTEX SHADER
//__________________________________________
struct VSOut {
    @builtin(position) position: vec4f, //position of each vertex in the screen
    @location(0) coords : vec2f, //coords that we transfer to fragment shader to compute rays and textures
};

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{   
    //4 vertex that cover the screen
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}



//__________________________________________
// RAY STRUCTURES
//__________________________________________
//ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
};


//intersection
struct HitInfo{
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,

    emission: vec3f, 
    diffuse: vec3f,
    specular: vec3f,

    shader: u32,
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

    (*hit).shader = 0u;
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
    L_i: vec3f, //light intensity
    w_i: vec3f, //light direction
    dist: f32, //distance to light
};

//shader constants
const SHADER_BASE : u32 = 0u;
const SHADER_LAMBERTIAN : u32 = 1u;
const SHADER_PHONG : u32 = 2u;
const SHADER_MIRROR : u32 = 3u;
const SHADER_REFRACT : u32 = 4u;



//__________________________________________
// GENERATE CAMERA RAY
//__________________________________________
fn get_camera_ray(ipcoords: vec2f) -> Ray
{
    //this function generates a ray from the camera through the pixel
    let q = uniforms.b1* ipcoords.x + uniforms.b2*ipcoords.y + uniforms.v*uniforms.cam_const; //we use cam_const here
    let r = Ray(uniforms.eye, normalize(q), 0.001, 10000.0); 
    return r;
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
fn sample_env(dir: vec3f) -> vec3f {
    let uv = direction_to_panorama_uv(dir);

    // dimensions de la texture
    let dims = textureDimensions(envMap); // vec2u
    let size_f = vec2f(f32(dims.x), f32(dims.y));

    // uv dans [0,1] → coord en pixels
    var st = uv * size_f;
    // on reste dans les bornes [0, size-1]
    st = clamp(st, vec2f(0.0), size_f - vec2f(1.0));

    let ij = vec2u(st);
    let texel = textureLoad(envMap, ij, 0);

    return texel.rgb;
}



//________________________
// FRESNEL R
//______________________

fn fresnel_R(cos_i : f32, cos_t : f32, eta : f32) -> f32 {
    // eta = ni / nt

    // total internal reflection
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

fn intersect_sphere(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, emission: vec3f, diffuse: vec3f, specular:vec3f, shade:u32) -> bool {
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
        (*hit).specular = specular;
        (*hit).shader = shade; 

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_triangle(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, face_index:u32, emission: vec3f, diffuse: vec3f, specular:vec3f, shade:u32) -> bool {
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
        (*hit).specular = specular;
        (*hit).shader = SHADER_LAMBERTIAN;

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
                if (intersect_triangle(r, hit, obj_idx, mat.emission, mat.color, vec3f(0.0), 1u)) {
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


fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, isPrimary: bool) -> bool {
    let r0 = *r;

    var r_box = r0;
    //if (isPrimary) {
    //    if (!intersect_aabb(&r_box)) { return false; }
    //}

    //TRIANGLES via BSP
    var hit_tri : HitInfo;
    reset_hit(&hit_tri, r_box.tmax);
    var r_tri = r_box;                 // copie dédiée
    let got_tri = intersect_bsp(&r_tri, &hit_tri);

    
    //spheres
    /*
    var hit_sphL : HitInfo;
    reset_hit(&hit_sphL, r_box.tmax);
    var r_sphL = r_box;                // copie dédiée
    let got_sphL = intersect_sphere(
        &r_sphL, &hit_sphL,
        vec3f(420.0, 90.0, 370.0),     // center (mets z négatif si ta caméra regarde +Z)
        90.0,                          // radius
        vec3f(0.0),                    // emission
        vec3f(0.8, 0.8, 0.8),                    // diffuse
        vec3f(0.0),                    // specular
        SHADER_MIRROR                 // shader
    );

    var hit_sphR : HitInfo;
    reset_hit(&hit_sphR, r_box.tmax);
    var r_sphR = r_box;                // copie dédiée
    let got_sphR = intersect_sphere(
        &r_sphR, &hit_sphR,
        vec3f(130.0, 90.0, 250.0),
        90.0,
        vec3f(0.0),
        vec3f(0.9, 0.9, 1.0),
        vec3f(0.0),
        SHADER_TRANSPARENT
    ); 
    hit_sphR.ior = 1.5; 
    
    

    var best_got = false;
    var best_hit : HitInfo;
    reset_hit(&best_hit, r_box.tmax);

    if (got_tri) { best_hit = hit_tri; best_got = true; }
    if (got_sphL && (!best_got || hit_sphL.dist < best_hit.dist)) { best_hit = hit_sphL; best_got = true; }
    if (got_sphR && (!best_got || hit_sphR.dist < best_hit.dist)) { best_hit = hit_sphR; best_got = true; }
    
    if (!best_got) {
        return false;
    }

    (*hit) = best_hit;

    if (best_hit.shader == SHADER_TRANSPARENT) {
        hit.extinction = vec3f(0.0, 0.15, 0.0);
    }

    (*r).tmax = (*hit).dist;
    return true;
    */
    if (!got_tri) {
        return false;
    }

    (*hit) = hit_tri;
    (*r).tmax = hit_tri.dist;

    return true;
}



//_________________________________________
// LIGHT SAMPLING
//_________________________________________

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

fn sample_directional_light(pos: vec3f) -> Light {
    //direction light : towards the scene
    let w_e = normalize(vec3f(-1.0, -1.0, -1.0));
    //radiance
    const L_e = vec3f(3.14159, 3.14159, 3.14159);
    return Light(L_e, w_e, 1e9);
}

fn sample_area_light(hitPos: vec3f, hitN: vec3f, t: ptr<function, u32>) -> Light {
    //random selection of a light triangle
    let n = arrayLength(&lightIndices);
    let eps_tri = rnd(t);
    let triIdx = lightIndices[u32(eps_tri * f32(n))];
    let face = meshFaces[triIdx];

    //get triangle vertices
    let v0 = attribs[face.x*2u + 0u].xyz;
    let v1 = attribs[face.y*2u + 0u].xyz;
    let v2 = attribs[face.z*2u + 0u].xyz;


    //random barycentric sampling
    let eps_1 = rnd(t);
    let eps_2 = rnd(t);
    let sqrt_eps1 = sqrt(eps_1);
    let alpha = 1.0 - sqrt_eps1;
    let beta = (1 - eps_2) * sqrt_eps1;
    let gamma = eps_2 * sqrt_eps1;
    let posL =  alpha * v0 + beta * v1 + gamma * v2;

    //light normal
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    var nL = normalize(cross(e1, e2));
    let area = 0.5 * length(cross(e1, e2));

    if (dot(nL, vec3f(0.0, -1.0, 0.0)) < 0.0) {
        nL = -nL;
    }

    //direction to light
    let wi = posL - hitPos;
    let dist2 = dot(wi, wi);
    let dist = sqrt(dist2);
    let dir = wi / dist;

    //cosines
    let cos_i = max(dot(hitN, dir), 0.0);
    let cos_o = max(dot(-dir, nL), 0.0);

    if (cos_i <= 0.0 || cos_o <= 0.0) {
        return Light(vec3f(0.0), dir, dist);
    }

    //shadow ray
    var shadowRay = Ray(hitPos + 0.0001 * hitN, dir, 0.001, dist - 0.001);
    var shadowHit: HitInfo;
    reset_hit(&shadowHit, 1e9);
    if (intersect_scene(&shadowRay, &shadowHit, false)) {
        return Light(vec3f(0.0), dir, dist); //in shadow
    }

    //radiance 
    let matIdx = face.w; 
    let mat = materials[matIdx];
    let Le = mat.emission;
    
    //pdf(x) = 1 / (n * area)
    //contribution = Le * cos_i * cos_o / (dist2 * pdf(x))
    let pdf = 1.0 / (f32(n) * area);
    let L_i = Le * (cos_o) / (dist2 * pdf);

    return Light(L_i, dir, dist);
}

//__________________________________________
// SHADING FUNCTIONS
//__________________________________________

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {

    var Lo = vec3f(0.0);

    // 1. ----- Direct lighting -----
    let Ld = sample_area_light((*hit).position, (*hit).normal, t);
    Lo += ((*hit).diffuse/3.14159) * (Ld.L_i * max(dot((*hit).normal, Ld.w_i), 0.0));

    // 2. ----- Emission -----
    if ((*hit).emit_flag == 1u) {
        Lo += (*hit).emission;
    }

    // 3. ----- Indirect illumination -----
    // Sample a cosine-weighted direction
    let v_local = sample_cosine_weighted(t);
    let wi = rotate_to_normal((*hit).normal, v_local);

    // Prepare next ray
    (*r).origin = (*hit).position + 0.001 * (*hit).normal;
    (*r).direction = wi;
    (*r).tmin = 0.001;
    (*r).tmax = 1e9;

    // Enable continuation of the path
    (*hit).continue_ray = true;
    
    // Indirect = diffuse (since pdf = cos/pi)
    return Lo;
}


fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let w_i = (*r).direction; //incoming direction w_i
    let n = normalize((*hit).normal); //normal at hit point

    let reflected_dir = normalize(w_i - 2.0  * dot(w_i, n)*n); //reflected direction

    (*r).origin = (*hit).position + reflected_dir * 0.005;
    (*r).direction = reflected_dir;
    (*r).tmin = 0.001;
    (*r).tmax = 1000.0;

    (*hit).continue_ray = true;
    (*hit).emit_flag = 1u;
    return vec3f(0.0); //white color for mirror
}

fn refract(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let n = normalize((*hit).normal);
    let d = normalize((*r).direction);

    var n_d = dot(n, d);
    var eta : f32;
    var normal = n;

    if (n_d < 0.0) {
        //from outside
        /*
        eta = (*hit).ior/1.5; //air(1.0) -> glass (1.5)
        (*hit).ior = 1.5;
        */
        eta = 1.0/1.5;
    } else {
        //from inside
        /*
        eta = (*hit).ior/ 1.0; //glass(1.5) -> air (1.0)
        (*hit).ior = 1.0;
        */
        eta = 1.5/1.0;
        normal = -n;
        n_d = -n_d;
    }

    //Snell formula
    let k = 1.0 - eta*eta*(1.0 - n_d*n_d);
    if (k < 0.0) {
        //total intern reflexion
        let reflected = normalize(d - 2.0*n_d*n);
        (*r).origin = (*hit).position;
        (*r).direction = reflected;
    } else {
        let refracted = eta*d - (eta*n_d + sqrt(k))*normal;
        (*r).origin = (*hit).position;
        (*r).direction = normalize(refracted);
    }
    (*r).tmin = 0.001;
    (*r).tmax = 1000.0;

    (*hit).continue_ray = true;
    return vec3f(0.0);
}

fn transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function,u32>) -> vec3f {

    let n = normalize(hit.normal);
    let d = normalize((*r).direction);

    var ni = 1.0;
    var nt = hit.ior;
    var normal = n;
    var cos_i = dot(-d, n);

    // ---------------------------------
    // Détection entrée / sortie
    // ---------------------------------
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

    // Fresnel
    let R = fresnel_R(cos_i, cos_t, eta);

    // Reflection ?
    if (rnd(t) < R || k < 0.0) {
        let refl = normalize(d - 2.0 * dot(d, normal) * normal);
        (*r).origin = hit.position + refl * 0.001;
        (*r).direction = refl;
        (*r).tmin = 0.001;
        (*r).tmax = 1e9;
        hit.continue_ray = true;
        return vec3f(0.0);
    }

    // -------------------------------------------------
    // TRANSMISSION
    // -------------------------------------------------

    let refr = normalize(eta * d + (eta*cos_i - cos_t)*normal);
    (*r).origin = hit.position + refr * 0.001;
    (*r).direction = refr;
    (*r).tmin = 0.001;
    (*r).tmax = 1e9;

    // ----------------------------------------
    // ABSORPTION SEULEMENT QUAND ON SORT DU VERRE
    // ----------------------------------------
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







fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_directional_light((*hit).position);
    let n = normalize((*hit).normal);
    let l = normalize(light.w_i);
    let v = normalize(-(*r).direction);
    let h = normalize(l+v);

    //Lambert (compute diffuse)
    let n_dot_l = max(dot(n,l), 0.0);
    let diffuse = ((*hit).diffuse / 3.14159) * light.L_i * n_dot_l;

    //phong (compute specular)
    let s: f32 = 42.0; //exponent
    let rho_s: vec3f = vec3f(0.1, 0.1, 0.1);
    let n_dot_h = max(dot(n, h), 0.0);
    let specular = rho_s * pow(n_dot_h, s) * light.L_i * n_dot_l;

    //shadow ray
    var shadow_ray = Ray((*hit).position, l, 0.001, light.dist - 0.001);
    var shadow_hit: HitInfo;
    reset_hit(&shadow_hit, 1e9);

    if (intersect_scene(&shadow_ray,&shadow_hit, false)) {
        return (*hit).emission; //if shadow 
    }

    return (*hit).emission + diffuse + specular;
}

fn glossy(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    //phong components
    let light = sample_point_light((*hit).position);
    let n = normalize((*hit).normal);
    let l = normalize(light.w_i);
    let v = normalize(-(*r).direction);
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let diffuse = ((*hit).diffuse / 3.14159) * light.L_i * n_dot_l;

    let s: f32 = 42.0;
    let rho_s: vec3f = vec3f(0.1, 0.1, 0.1);
    let n_dot_h = max(dot(n, h), 0.0);
    let specular = rho_s * pow(n_dot_h, s) * light.L_i;

    var phongColor = (*hit).emission + diffuse + specular;

    //refraction component
    let refractColor = refract(r, hit);

    //Combine 
    return phongColor + refractColor;

}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t:ptr<function, u32>) -> vec3f {
    switch u32(uniforms.objectShader) {
        case SHADER_BASE: {return hit.diffuse; }     // couleur de base
        case SHADER_LAMBERTIAN: {return lambertian(r, hit, t);}
        case SHADER_PHONG: {return phong(r, hit);}
        case SHADER_MIRROR: {return mirror(r, hit);}
        default: {return lambertian(r, hit, t);}
    }

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

    return vec3f(x, y, z); // dans espace tangent (normal = +Z)
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
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> @location(0) vec4f
{   
    
    /*
    let eye = uniforms.eye;
    
    let uv = 0.5 * (coords + vec2f(1.0, 1.0));
    let texcolor = textureSample(my_texture, my_sampler, uv);
    return texcolor;
    */

    let ipcoords = coords*0.5; //normalize

    //random seed
    let launch_idx = u32(fragcoord.y) * u32(uniforms.width) + u32(fragcoord.x);
    var t = tea(launch_idx, u32(uniforms.frame));

    var final_color = vec3f(0.0, 0.0, 0.0);

    //number of subdivisions
    let n = i32(uniforms.subdivs * uniforms.subdivs);
    let gamma = uniforms.gamma;

    for (var k = 0; k < n; k++) {
        //jitter
        let offset = jitter[k];
        let jittered = ipcoords + offset;

        var r = get_camera_ray(jittered);
        /*DEBUG
        if (!intersect_aabb(&r)) {
            //red = ray rejected by AABB
            return vec4f(1.0, 0.0, 0.0, 1.0);
        } else {
            //green = ray passes AABB
            return vec4f(0.0, 1.0, 0.0, 1.0);
        }
        */
        var color = vec3f(0.0, 0.0, 0.0);
        var throughput = vec3f(1.0);

        for (var depth = 0; depth < 5; depth++){
            var hit: HitInfo;
            reset_hit(&hit, 1e9);

            let hit_something = intersect_scene(&r, &hit, depth == 0);
            if (depth > 0) {
                hit.emit_flag = 0u;
            }
            if (!hit_something) {
                let env = sample_env(r.direction);
                color += throughput * env;
                break;
            }


            hit.throughput = throughput;     // préparer throughput pour le shader
            let shaded = shade(&r, &hit, &t);
            color += hit.throughput * shaded;

            if (
                hit.shader != SHADER_MIRROR)
            {
                throughput = hit.throughput * hit.diffuse;
            }
            else {
                throughput = hit.throughput;
            }
            if (!hit.continue_ray) { break; }

        }
        final_color += color;
    }

    final_color /= f32(n);

    //random jitter 
    let jitter_offset = (vec3f(rnd(&t), rnd(&t), rnd(&t)) - vec3f(0.5)) / f32(uniforms.height);

    //get previous color if progressive
    

    return vec4f(final_color, 1.0);
}
