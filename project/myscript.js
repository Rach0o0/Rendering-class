"use strict";

//entry point
window.onload = function() { main(); }

//HDR environment map loading
async function loadHDRToTexture(device, url) {
    return new Promise((resolve, reject) => {
        const hdr = new HDRImage();

        hdr.onload = () => {
            const width = hdr.width;
            const height = hdr.height;

            const dataRGBE = hdr.dataRGBE;

            const texture = device.createTexture({
                size: { width, height, depthOrArrayLayers: 1 },
                format: "rgba8uint",
                usage: GPUTextureUsage.TEXTURE_BINDING |
                       GPUTextureUsage.COPY_DST,
            });

            device.queue.writeTexture(
                { texture },
                dataRGBE,
                {
                    bytesPerRow: width * 4,
                    rowsPerImage: height,
                },
                { width, height, depthOrArrayLayers: 1 }
            );

            resolve({ texture, width, height, dataRGBE });
        };

        hdr.onerror = (e) => {
            console.error("Failed to load HDR:", e);
            reject(e);
        };

        hdr.src = url;
    });
}

//__________________________________________
// HDR utilities 
//__________________________________________
function decodeRGBE_toLinear(r, g, b, e) {
  if (e === 0) return [0, 0, 0];
  const f = Math.pow(2, (e - 128) - 8);
  return [r * f, g * f, b * f];
}

function findSunDirectionFromRGBE(dataRGBE, width, height) {
  let best = -1;
  let bestX = 0, bestY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = 4 * (y * width + x);
      const r = dataRGBE[i];
      const g = dataRGBE[i + 1];
      const b = dataRGBE[i + 2];
      const e = dataRGBE[i + 3];

      const [R, G, B] = decodeRGBE_toLinear(r, g, b, e);
      const lum = 0.2126 * R + 0.7152 * G + 0.0722 * B;

      if (lum > best) {
        best = lum;
        bestX = x;
        bestY = y;
      }
    }
  }

  const u = (bestX + 0.5) / width;
  const v = (bestY + 0.5) / height;

  const theta = (u - 0.5) * 2.0 * Math.PI;
  const ydir = Math.sin((0.5 - v) * Math.PI);
  const cosLat = Math.sqrt(1.0 - ydir * ydir);

  return [
    Math.cos(theta) * cosLat,
    ydir,
    Math.sin(theta) * cosLat
  ];
}


async function main()
{
    //______________________________________
    // INITIALIZATION
    //______________________________________
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice({
        requiredFeatures : ['timestamp-query'],
        requiredLimits:{
            maxStorageBuffersPerShaderStage: 10
        }
    });

    const timing = new TimingHelper(device);

    //_____________________________________
    // LOAD 3D OBJECT
    //_____________________________________
    const obj = await readOBJFile("objects/bunny.obj", 1, true);


    // ------------------------------------------
    // scale 
    // ------------------------------------------
    const stride = 8; // x y z nx ny nz u v
    const DRAGON_SCALE = 20.0;
    for (let i = 0; i < obj.attribs.length; i += stride) {
        obj.attribs[i + 0] *= DRAGON_SCALE;
        obj.attribs[i + 1] *= DRAGON_SCALE;
        obj.attribs[i + 2] *= DRAGON_SCALE;
    }

    const buffers = {};
    build_bsp_tree(obj, device, buffers);
    

    function computeAABBFromAttribs(attribs) {
        let min = vec3( Infinity,  Infinity,  Infinity);
        let max = vec3(-Infinity, -Infinity, -Infinity);

        const stride = 8;

        for (let i = 0; i < attribs.length; i += stride) {
            let x = attribs[i + 0];
            let y = attribs[i + 1];
            let z = attribs[i + 2];

            min[0] = Math.min(min[0], x);
            min[1] = Math.min(min[1], y);
            min[2] = Math.min(min[2], z);

            max[0] = Math.max(max[0], x);
            max[1] = Math.max(max[1], y);
            max[2] = Math.max(max[2], z);
        }

        return { min, max };
    }

    const aabb = computeAABBFromAttribs(obj.attribs);

    //________________________________________________
    // MATERIAL BUFFER
    //_______________________________________________
    const numMaterials = obj.materials.length;
    const materialData = new Float32Array(numMaterials * 8); 

    for (let i = 0; i < numMaterials; i++) {
        const mat = obj.materials[i];

        materialData[i * 8 + 0] = mat.color.r;
        materialData[i * 8 + 1] = mat.color.g;
        materialData[i * 8 + 2] = mat.color.b;
        materialData[i * 8 + 3] = 0.0;

        if (mat.emission) {
            materialData[i * 8 + 4] = mat.emission.r;
            materialData[i * 8 + 5] = mat.emission.g;
            materialData[i * 8 + 6] = mat.emission.b;
        } else {
            materialData[i * 8 + 4] = 0.0;
            materialData[i * 8 + 5] = 0.0;
            materialData[i * 8 + 6] = 0.0;
        }
        materialData[i * 8 + 7] = 0.0; 
    }

    buffers.materials = device.createBuffer({
        size: materialData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(buffers.materials, 0, materialData);

    //____________________________________________
    // LIGHT INDICES BUFFER
    //____________________________________________
    let li = obj.light_indices;
    if (li.length === 0) li = new Uint32Array([0]);

    buffers.lightIndices = device.createBuffer({
        size: li.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(buffers.lightIndices, 0, li);

    //______________________________________
    // SETUP CANVAS
    //______________________________________
    const canvas = document.getElementById('my-canvas');
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({
        device: device,
        format: canvasFormat,
    });

    //______________________________________
    // TEXTURES
    //______________________________________
    const textures = {};
    textures.width = canvas.width;
    textures.height = canvas.height;

    textures.renderDst = device.createTexture({
        size: [textures.width, textures.height],
        format: 'rgba32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.TEXTURE_BINDING
    });

    //______________________________________
    // LOAD HDR ENV MAP
    //______________________________________
    const { texture: envTexture, width: envW, height: envH, dataRGBE } =
        await loadHDRToTexture(device, "HDR_blue_nebulae-1.hdr.png");

    const sunDir = findSunDirectionFromRGBE(dataRGBE, envW, envH);


    const envSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    //______________________________________
    // LOAD SHADER
    //______________________________________
    const wgslfile = document.getElementById('wgsl').src;
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({
        code: wgslcode
    });

    //______________________________________
    // CREATE RENDER PIPELINE
    //______________________________________
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 1,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 2,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 3,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 4,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 5,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 6,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 7,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 8,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 9,  visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }, 
            { binding: 11, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "uint"} },
            { binding: 12, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });
    
    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: wgsl, entryPoint: 'main_vs', },
        fragment: {
            module: wgsl,
            entryPoint: 'main_fs',
            targets: [{ format: canvasFormat }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    //______________________________________
    // UNIFORM BUFFER
    //______________________________________
    let bytelength = 12 * 16;
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    //________________________________________
    // CREATE JITTER BUFFER
    //________________________________________
    let jitter = new Float32Array(200);

    const jitterBuffer = device.createBuffer({
        size: jitter.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });

    function compute_jitters(jitter, pixelsize, subdivs) {
        const step = pixelsize / subdivs;
        if (subdivs < 2) {
            jitter[0] = 0.0;
            jitter[1] = 0.0;
        } else {
            for (let i = 0; i < subdivs; ++i) {
                for (let j = 0; j < subdivs; ++j) {
                    const idx = (i * subdivs + j) * 2;
                    jitter[idx]     = (Math.random() + j) * step - pixelsize * 0.5;
                    jitter[idx + 1] = (Math.random() + i) * step - pixelsize * 0.5;
                }
            }
        }
    }

    let subdivs = 1;

    function updateJitters(){
        const pixelsize = 1.0 / canvas.height;
        compute_jitters(jitter, pixelsize, subdivs);
        device.queue.writeBuffer(jitterBuffer, 0, jitter);
    }

    //______________________________________
    // ASSOCIATE BUFFER AND TEXTURE TO PIPELINE
    //______________________________________
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0,  resource: { buffer: uniformBuffer }},
            { binding: 1,  resource: { buffer: jitterBuffer }},
            { binding: 2,  resource: { buffer: buffers.attribs } },
            { binding: 3,  resource: { buffer: buffers.indices } },
            { binding: 4,  resource: { buffer: buffers.materials } },
            { binding: 5,  resource: { buffer: buffers.bspTree } },
            { binding: 6,  resource: { buffer: buffers.treeIds } }, 
            { binding: 7,  resource: { buffer: buffers.bspPlanes } },
            { binding: 8,  resource: { buffer: buffers.aabb } }, 
            { binding: 9,  resource: { buffer: buffers.lightIndices } },
            { binding: 11, resource: envTexture.createView() },
            { binding: 12, resource: envSampler },  
        ],
    });
    
    //______________________________________
    // PARAMETERS
    //______________________________________
    const orbitCenter = vec3(
        0.5 * (aabb.min[0] + aabb.max[0]),
        0.5 * (aabb.min[1] + aabb.max[1]),
        0.5 * (aabb.min[2] + aabb.max[2])
    );

    
    let eye = vec3(
        orbitCenter[0],
        orbitCenter[1] + 1.5,
        orbitCenter[2] + 6.0
    );

    let p = vec3(
        orbitCenter[0],
        orbitCenter[1] + 0.8,
        orbitCenter[2]
    );


    let u   = vec3(0.0, 1.0, 0.0);
    let cam_const = 1.0;
    let gamma = 2.2;
    let frame = 0;
    let progressive = true;
    const envScale = 0.10;
    const sunRadiance = [15.0, 14.0, 12.0];
    let aperture = 0.0; 
    let focusDist = 6.0;  
    let autoFocus = false;
    let focusTime = 0.0;
    let dolly = false;
    let dollyTime = 0.0;
    let dollyDuration = 2.5; 

    let dollyDir = 1.0;

    let eyeStart, pStart, camStart;


    let orbit = false;
    let orbitAngle = 0.0;
    let orbitRadiusX = 3.5;
    let orbitRadiusZ = 6.0;
    let orbitHeight  = 1.8;
    let orbitSpeed   = 0.0015;




    //______________________________________
    // FILLS THE UNIFORM BUFFER
    //______________________________________
    function writeCameraUniforms(){
        const aspect = canvas.width / canvas.height;

        let v = normalize(subtract(p, eye));
        let b1 = normalize(cross(v, u));
        let b2 = cross(b1, v);

        let values = [
            aspect, cam_const, gamma, subdivs,
            ...eye, 0.0,
            ...b1,  0.0,
            ...b2,  0.0,
            ...v,   0.0,
            0.0, 0.0, canvas.width, canvas.height,
            frame, progressive ? 1 : 0, 0.0, 0.0,
            ...sunDir, 0.0,
            ...sunRadiance, envScale, 
            aperture, focusDist, 0.0, 0.0,
            ...orbitCenter, 0.0,
        ];

        const uniformData = new Float32Array(values);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    }

    //_______________________________________
    // RENDER LOOP
    //_______________________________________
    function render(){
        if (progressive) frame++;
        else frame = 0;

        if (autoFocus) {
            focusTime += 0.01;

            let t = 0.5 + 0.5 * Math.sin(focusTime);

            let focusNear = 3.0;
            let focusFar  = 10.0;

            focusDist = focusNear + t * (focusFar - focusNear);
        }

        if (dolly) {
            dollyTime += 1.0 / 60.0;

            let t = Math.min(dollyTime / dollyDuration, 1.0);

            t = t * t * (3.0 - 2.0 * t);

            let viewDir = normalize(subtract(pStart, eyeStart));

            let dollyDist = 1.5;

            let signedT = dollyDir * t;

            eye = add(eyeStart, scale(signedT * dollyDist, viewDir));
            p   = add(pStart,   scale(signedT * dollyDist, viewDir));

            cam_const = camStart * (1.0 - 0.25 * signedT);

            if (t >= 1.0) {
                dolly = false;
            }
        }


        if (orbit) {
            orbitAngle += orbitSpeed;

            let x = orbitCenter[0] + orbitRadiusX * Math.cos(orbitAngle);
            let z = orbitCenter[2] + orbitRadiusZ * Math.sin(orbitAngle);

            let y = orbitCenter[1] + orbitHeight;

            eye = vec3(x, y, z);

            p = add(orbitCenter, vec3(0.0, 0.15, 0.0));
        }



        writeCameraUniforms();

        const encoder = device.createCommandEncoder();

        const pass = timing.beginRenderPass(encoder, {
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
        }]
        });


        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();

        device.queue.submit([encoder.finish()]);

        timing.getResult().then(durationNs => {
            const ms = durationNs / 1e6;
            document.getElementById("frametime").textContent =
            `Frame time: ${ms.toFixed(2)} ms`;
        });

        requestAnimationFrame(render);
    }
    
    //_______________________________________
    // UI
    //_______________________________________

    window.addEventListener('wheel', (e) => {
        cam_const *= 1.0 + 2.5e-4 * e.deltaY;
        requestAnimationFrame(render);
    }, {passive:false});

    document.getElementById('zoom').addEventListener('input', (e) => {
        cam_const = parseFloat(e.target.value);
        requestAnimationFrame(render);
    });

    document.getElementById("gamma-slider").addEventListener("input", (e) => {
        gamma = parseFloat(e.target.value);
        frame = 0;
        requestAnimationFrame(render);
    });

    document.getElementById("subdivs").addEventListener("input", (e) => {
        subdivs = parseInt(e.target.value);
        updateJitters();
        frame = 0;
        requestAnimationFrame(render);
    });

    document.getElementById("progressive").addEventListener("change", (e) => {
        progressive = e.target.checked;
        frame = 0;
    });

    document.getElementById("aperture").addEventListener("input", (e) => {
        aperture = parseFloat(e.target.value);
        frame = 0;
    });

    document.getElementById("focus").addEventListener("input", (e) => {
        focusDist = parseFloat(e.target.value);
        frame = 0;
    });
    document.getElementById("autoFocus").addEventListener("change", (e) => {
        autoFocus = e.target.checked;
        frame = 0;
    });
    document.getElementById("dolly").addEventListener("change", (e) => {
        dolly = true;
        dollyTime = 0.0;

        dollyDir = e.target.checked ? 1.0 : -1.0;

        eyeStart = vec3(eye[0], eye[1], eye[2]);
        pStart   = vec3(p[0],   p[1],   p[2]);
        camStart = cam_const;
    });
    document.getElementById("orbit").addEventListener("change", (e) => {
        orbit = e.target.checked;
    });




    //_______________________________________________
    // FIRST FRAME
    //_____________________________________________
    updateJitters();
    requestAnimationFrame(render);
}
