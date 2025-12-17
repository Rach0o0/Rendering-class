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
    sphereShader: f32, mattShader: f32, _pad5: vec2f,
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
const SHADER_LAMBERTIAN : u32 = 1u;
const SHADER_PHONG : u32 = 2u;
const SHADER_MIRROR : u32 = 3u;
const SHADER_REFRACT : u32 = 4u;
const SHADER_GLOSSY : u32 = 5u;

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
        (*hit).shader = shade; 

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
    // 0) Sauvegarde du rayon d’entrée
    let r0 = *r;

    // 1) Clipper dans l’AABB pour les rayons primaires (sans toucher r)
    var r_box = r0;
    if (isPrimary) {
        if (!intersect_aabb(&r_box)) { return false; }
    }

    // --- 2) TRIANGLES via BSP ---
    var hit_tri : HitInfo;
    reset_hit(&hit_tri, r_box.tmax);
    var r_tri = r_box;                 // copie dédiée
    let got_tri = intersect_bsp(&r_tri, &hit_tri);

    // --- 3) SPHÈRE MIROIR ---
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
        SHADER_MIRROR                  // shader
    );

    // --- 4) SPHÈRE VERRE ---
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
        SHADER_REFRACT
    );

    // --- 5) Sélection du hit le plus proche ---
    var best_got = false;
    var best_hit : HitInfo;
    reset_hit(&best_hit, r_box.tmax);

    if (got_tri) { best_hit = hit_tri; best_got = true; }
    if (got_sphL && (!best_got || hit_sphL.dist < best_hit.dist)) { best_hit = hit_sphL; best_got = true; }
    if (got_sphR && (!best_got || hit_sphR.dist < best_hit.dist)) { best_hit = hit_sphR; best_got = true; }

    if (best_got) {
        (*hit) = best_hit;
        // Très important : borne le rayon appelant pour la suite (ombre, rebonds…)
        (*r).tmax = best_hit.dist;
    }
    return best_got;
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

fn sample_area_light(hitPos: vec3f, hitN: vec3f) -> Light {
    var L_accum = vec3f(0.0);
    var dir_accum = vec3f(0.0);
    var total_weight = 0.0;

    for (var i: u32 = 0u; i < arrayLength(&lightIndices); i = i + 1u) {
        let triIdx = lightIndices[i];
        let face = meshFaces[triIdx];
        //interleaved positions
        let v0 = attribs[face.x*2u + 0u].xyz;
        let v1 = attribs[face.y*2u +0u].xyz;
        let v2 = attribs[face.z*2u + 0u].xyz;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        var nL = normalize(cross(e1, e2));
        let area = 0.5 * length(cross(e1, e2));
        let center = (v0 + v1 + v2) / 3.0;
        
        if (dot(nL, vec3f(0.0, -1.0, 0.0)) < 0.0) {
            nL = -nL;
        }
        
        let wi = center - hitPos;
        let dist2 = dot(wi, wi);
        let dist = sqrt(dist2);
        let dir = wi / dist;

        let cos_i = max(dot(hitN, dir), 0.0);
        if (cos_i <= 0.0) {
            //light is behind surface
            continue;
        }
        
        let cos_o = max(dot(-dir, nL), 0.0);
        if (cos_o <= 0.0) {
            //triangle emits in the other way
            continue;
        }

        //shadow ray
        var shadowRay = Ray(hitPos + 0.001 *hitN, dir, 0.001, dist - 0.001);
        var shadowHit: HitInfo;
        reset_hit(&shadowHit, 1e9);

        let blocked = intersect_scene(&shadowRay, &shadowHit, false);
        if (blocked && shadowHit.emission.x <= 0.0 && shadowHit.emission.y <= 0.0 && shadowHit.emission.z <= 0.0) { continue; }

        let matIdx = face.w;
        let mat = materials[matIdx];
        let Le = mat.emission;

        let dE = 3.0 * Le * (cos_i * cos_o / (3.14159 * dist2)) * area;
        /*
        if (i == 0u) {
            //debug: encode L contribution of the first triangle
            return Light(vec3f(cos_i, cos_o, 0.0), dir, dist);
        }*/
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
    let lo = Le + (kd / 3.14159) * light.L_i * n_dot_wi; 
    (*hit).continue_ray = true; 
    return lo; }

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let w_i = (*r).direction; //incoming direction w_i
    let n = normalize((*hit).normal); //normal at hit point

    let reflected_dir = normalize(w_i - 2.0  * dot(w_i, n)*n); //reflected direction

    (*r).origin = (*hit).position;
    (*r).direction = reflected_dir;
    (*r).tmin = 0.001;
    (*r).tmax = 1000.0;

    (*hit).continue_ray = true; //flag to continue the ray path
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
    var shadow_hit = HitInfo(false, 1e9, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u, false, 1.0, vec2f(0.0), 0u);

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

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    if ((*hit).shader == SHADER_MIRROR) {
        return mirror(r, hit);
    } else if ((*hit).shader == SHADER_REFRACT) {
        return refract(r, hit);
    } else {
        return lambertian(r, hit);
    }
}




//__________________________________________
// FRAGMENT SHADER
//__________________________________________
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{   
    
    /*
    let eye = uniforms.eye;
    
    let uv = 0.5 * (coords + vec2f(1.0, 1.0));
    let texcolor = textureSample(my_texture, my_sampler, uv);
    return texcolor;
    */

    let ipcoords = coords*0.5; //normalize

    var hit: HitInfo;
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
        var hit: HitInfo;
        var color = vec3f(0.0, 0.0, 0.0);

        
        for (var depth = 0; depth < 5; depth++){
            var hit: HitInfo;
            reset_hit(&hit, 1e9);
            let hit_something = intersect_scene(&r, &hit, true);

            if (!hit_something && depth == 0) {
                //no intersection -> background color
                color += vec3f(0.1, 0.1, 0.1);
                break;
            }
            else if (!hit_something) {
                //no intersection for secondary rays
                break;
            }

            //shading 
            let shaded = shade(&r, &hit);
            color += shaded;

            if (!hit.continue_ray) {
                break;
            }
        }

        final_color += color;
    }

    final_color /= f32(n);

    let corrected = pow(final_color, vec3f(1.0 / gamma));

    return vec4f(corrected, 1.0);
}
