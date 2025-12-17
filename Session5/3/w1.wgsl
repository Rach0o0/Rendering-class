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
    useTexture: f32, filterFlag: f32, repeatFlag: f32, texScale: f32
}; 



//__________________________________________
// BINDINGS
//__________________________________________

@group(0) @binding(0) var<uniform> uniforms : Uniforms; //all info on uniforms from js
@group(0) @binding(1) var my_texture: texture_2d<f32>; //texture
@group(0) @binding(2) var<storage> jitter: array<vec2f>;
@group(0) @binding(3) var<storage> vPositions : array<vec3f>;
@group(0) @binding(4) var<storage> meshFaces : array<vec3u>;
@group(0) @binding(5) var<storage> vNormals : array<vec3f>;


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

    ambient: vec3f, 
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

    (*hit).ambient = vec3f(0.0);
    (*hit).diffuse = vec3f(0.0);
    (*hit).specular = vec3f(0.0);

    (*hit).shader = 0u;
    (*hit).continue_ray = false;
    (*hit).ior = 1.0;
    (*hit).texcoords = vec2f(0.0);
}


//___________________________________
// ORTHONORMAL BASIS (used for ground plane)
//___________________________________

struct Onb {
    tangent: vec3f,
    binormal: vec3f,
    normal: vec3f,
};

//fixed basis for ground
const plane_onb = Onb(
    vec3f(-1.0, 0.0, 0.0), //tangent
    vec3f(0.0, 0.0, 1.0), //binormal
    vec3f(0.0, 1.0, 0.0) //normal
);

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
    let r = Ray(uniforms.eye, normalize(q), 0.001, 1000.0); 
    return r;
}

//_____________________________
// TEXTURE FUNCTIONS
//_____________________________

//address mode : repeat or clamp
fn address_uv(uv : vec2f, repeat: bool) -> vec2f {
    if (repeat) {
        return fract(uv);
    }
    else {
        return clamp(uv, vec2f(0.0), vec2f(1.0));
    }
}

//nearest-neighbour
fn texture_nearest(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let size = vec2f(textureDimensions(texture, 0));
    let st = address_uv(texcoords, repeat);
    let uv = st * size;
    let U = clamp(i32(floor(uv.x + 0.5)), 0, i32(size.x - 1)); //nearest pixel w.r.t x
    let V = clamp(i32(floor(uv.y + 0.5)), 0, i32(size.y - 1)); //nearest w.r.t y
    let texel = textureLoad(texture, vec2<i32>(U,V), 0);
    return texel.rgb;
}

//linear
fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let size = vec2f(textureDimensions(texture, 0));
    let st = address_uv(texcoords, repeat);
    let uv = st * size;

    //coord texel left bottom
    let U = i32(floor(uv.x));
    let V = i32(floor(uv.y));

    let cx = uv.x - f32(U);
    let cy = uv.y - f32(V);

    //neighbors
    let c00 = textureLoad(texture, vec2<i32>(U, V), 0).rgb;
    let c10 = textureLoad(texture, vec2<i32>(U+1, V), 0).rgb;
    let c01 = textureLoad(texture, vec2<i32>(U, V+1), 0).rgb;
    let c11 = textureLoad(texture, vec2<i32>(U+1, V+1), 0).rgb;

    let c0 = mix(c00, c10, cx);
    let c1 = mix(c01, c11, cx);

    return mix(c0, c1, cx);
}

//________________________________________
// INTERSECTIONS
//________________________________________

fn intersect_plane(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, position: vec3f, onb:Onb, ambient: vec3f, diffuse:vec3f, specular:vec3f, shade:u32) -> bool {
    /*
    plane equation :(p - p0) . n = 0 
    ray equation : r(t) = o + t*w
    t = ((p0 - o).n) / (w.n)
    */
    let denom = dot((*r).direction, onb.normal); // w.n
    if (abs(denom) < 1e-6) {
        return false; //parallel ray (no intersection)
    }

    let t = dot(position - (*r).origin, onb.normal) / denom; //t = ((p0 - o).n) / (w.n)

    if (t > (*r).tmin && t < (*r).tmax && t < (*hit).dist) {
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = r.origin + t * r.direction;
        (*hit).normal = normalize(onb.normal);
        (*hit).ambient = ambient;
        (*hit).diffuse = diffuse;
        (*hit).specular = specular;
        (*hit).shader = shade; //plane shader

        //compute coords of the texture
        let scale = uniforms.texScale * 0.2;
        let rel = (*hit).position - position;
        let u = dot(rel, onb.tangent)* scale;
        let v = dot(rel, onb.binormal) * scale;
        (*hit).texcoords = vec2f(u,v);

        (*r).tmax = t; //update ray tmax to the intersection point
        return true;
    }
    return false;
}

fn intersect_sphere(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, ambient: vec3f, diffuse: vec3f, specular:vec3f, shade:u32) -> bool {
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
        (*hit).ambient = ambient;
        (*hit).diffuse = diffuse;
        (*hit).specular = specular;
        (*hit).shader = shade; 

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_triangle(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, face_index:u32, ambient: vec3f, diffuse: vec3f, specular:vec3f, shade:u32) -> bool {
    //take 3 indexed vertices
    let face = meshFaces[face_index];
    let v0 = vPositions[face.x];
    let v1 = vPositions[face.y];
    let v2 = vPositions[face.z];

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

    //normal 
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
        (*hit).ambient = ambient;
        (*hit).diffuse = diffuse;
        (*hit).specular = specular;
        (*hit).shader = shade; 

        (*r).tmax = t; 
        return true;
    }
    return false;
}

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) ->bool {
    //plane
    let plane_pos = vec3f(0.0, 0.0, 0.0);
    let plane_normal = vec3f(0.0, 1.0, 0.0);
    let plane_ambient = vec3f(0.1, 0.7, 0.0) * 0.1;
    let plane_diffuse = vec3f(0.1, 0.7, 0.0) * 0.9;
    let plane_specular = vec3f(0.0, 0.0, 0.0);
    let plane_shader = u32(uniforms.mattShader);
    if (intersect_plane(r, hit, plane_pos, plane_onb, plane_ambient, plane_diffuse, plane_specular, plane_shader)) {
        (*hit).object_id = 0u;
    }
    
    //triangles
    let tri_color = vec3f(0.9);  // couleur constante pour tout le teapot
    let tri_ambient  = tri_color * 0.1;
    let tri_diffuse  = tri_color * 0.9;
    let tri_specular = vec3f(0.0);
    let tri_shader   = u32(uniforms.mattShader);

    for (var i: u32 = 0u; i < arrayLength(&meshFaces); i = i + 1u) {
        if (intersect_triangle(r, hit, i, tri_ambient, tri_diffuse, tri_specular, tri_shader)) {
            (*hit).object_id = 2u;
        }
    }

    return (*hit).has_hit;
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
    return Light(L_e, -w_e, 1e9);
}

//__________________________________________
// SHADING FUNCTIONS
//__________________________________________

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_point_light((*hit).position);
    let n_dot_wi = max(dot((*hit).normal, light.w_i), 0.0);

    //diffuse / ambient
    var kd = (*hit).diffuse;
    var ka = (*hit).ambient;

    //texture flags
    let use_tex = (uniforms.useTexture > 0.5);
    let repeat = (uniforms.repeatFlag > 0.5);

    //if texture activated
    if (use_tex && ((*hit).object_id == 0u)) { //only for plane
        if(uniforms.filterFlag < 0.5) { // nearest
            let texcolor = texture_nearest(my_texture, (*hit).texcoords, repeat);
            kd = texcolor * 0.9;
            ka = texcolor * 0.1;
        } else { //linear
            let texcolor = texture_linear(my_texture, (*hit).texcoords, repeat);
            kd = texcolor * 0.9;
            ka = texcolor * 0.1;
        }
    }

    //shadow ray
    var shadow_ray = Ray((*hit).position, light.w_i, 0.001, light.dist - 0.001);
    var shadow_hit = HitInfo(false, 1e9, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u, false, 1.0, vec2f(0.0), 0u);
    
    //check if the shadow ray intersects any object
    if (intersect_scene(&shadow_ray, &shadow_hit)){
        return ka; //in shadow, return only ambient
    }
    

    //not in shadow, return ambient + diffuse
    let lo = ka + (kd / 3.14159) * light.L_i * n_dot_wi;
    return lo;
}

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

    if (intersect_scene(&shadow_ray,&shadow_hit)) {
        return (*hit).ambient; //if shadow 
    }

    return (*hit).ambient + diffuse + specular;
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

    var phongColor = (*hit).ambient + diffuse + specular;

    //refraction component
    let refractColor = refract(r, hit);

    //Combine 
    return phongColor + refractColor;

}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{
    switch (*hit).shader {
        case SHADER_LAMBERTIAN { 
            return lambertian(r, hit); 
        }
        case SHADER_MIRROR {
            return mirror(r, hit);
        }
        case SHADER_REFRACT{
            return refract(r,hit);
        }
        case SHADER_PHONG {
            return phong(r,hit);
        }
        case SHADER_GLOSSY {
            return glossy(r,hit);
        }
        case default { 
            return (*hit).ambient + (*hit).diffuse;
        }
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
        var hit: HitInfo;
        var color = vec3f(0.0, 0.0, 0.0);

        
        for (var depth = 0; depth < 5; depth++){
            var hit: HitInfo;
            reset_hit(&hit, 1e9);
            let hit_something = intersect_scene(&r, &hit);

            if (!hit_something) {
                //no intersection -> background color
                color += vec3f(0.1, 0.3, 0.6);
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
