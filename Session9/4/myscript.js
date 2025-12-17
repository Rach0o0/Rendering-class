"use strict";

//when the page load, we call main
window.onload = function() { main(); }

async function loadHDRToTexture(device, url) {
    return new Promise((resolve, reject) => {
        const hdr = new HDRImage();

        hdr.onload = () => {
            const width = hdr.width;
            const height = hdr.height;

            // RGBE 8-bit (obligatoire pour WGSL u32)
            const dataRGBE = hdr.dataRGBE;

            const texture = device.createTexture({
                size: { width, height, depthOrArrayLayers: 1 },
                format: "rgba8uint",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
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

        hdr.onerror = reject;
        hdr.src = url;
    });
}


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

    //_____________________________________
    // LOAD 3D OBJECT
    //_____________________________________
    const obj = await readOBJFile("objects/teapot.obj", 1, true);
    
    const buffers = {};
    build_bsp_tree(obj, device, buffers); // --> buffers.bspTree, buffers.treeIds, buffers.bspPlanes, buffers.aabb
    console.log("Interleaved BSP buffers:", buffers);
    console.log("AABB Min:", buffers.aabb_min);
    console.log("AABB Max:", buffers.aabb_max);

    //________________________________________________
    // MATERIAL BUFFER
    //_______________________________________________
    const numMaterials = obj.materials.length;
    const materialData = new Float32Array(numMaterials * 8); 

    for (let i = 0; i < numMaterials; i++) {
        const mat = obj.materials[i];
        //diffuse color(kd)
        materialData[i * 8 + 0] = mat.color.r;
        materialData[i * 8 + 1] = mat.color.g;
        materialData[i * 8 + 2] = mat.color.b;
        materialData[i * 8 + 3] = 0.0; // padding

        //emission(ka)
        if (mat.emission) {
            materialData[i * 8 + 4] = mat.emission.r;
            materialData[i * 8 + 5] = mat.emission.g;
            materialData[i * 8 + 6] = mat.emission.b;
        } else {
            materialData[i * 8 + 4] = 0.0;
            materialData[i * 8 + 5] = 0.0;
            materialData[i * 8 + 6] = 0.0;
        }
        materialData[i * 8 + 7] = 0.0; //padding
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

    console.log("Light indices array:", obj.light_indices);
    for (let i = 0; i < obj.light_indices.length; i++) {
        console.log(`Light triangle ${i}:`, obj.light_indices[i]);
    }

    console.log("Loaded materials:");
    for (let i = 0; i < numMaterials; i++) {
        const mat = obj.materials[i];
        console.log(i, "color:", mat.color, "emission:", mat.emission);
    }

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
    // PINGPONG TEXTURES
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
    // LOAD HDR ENV MAP (via hdrpng.js)
    //______________________________________
    const { texture: envTexture, width: envW, height: envH, dataRGBE } =
        await loadHDRToTexture(device, "symmetrical_garden_02_4k.RGBE.PNG");

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
    let bytelength = 9 * 16; // Buffers are allocated in vec4 chunks
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
    const eye = vec3(0.0, 40.0, 300.0);
    const p   = vec3(0.0, 30.0, 0.0);
    const u   = vec3(0.0, 1.0, 0.0);
    let cam_const = 1.0;
    let gamma = 2.2;
    let frame = 0;
    let progressive = true;
    let bgFlag = 0.0;
    let mattShader = 1.0;
    const envScale = 0.12;
    const sunRadiance = [25.0, 23.0, 18.0];

    //______________________________________
    // FILLS THE UNIFORM BUFFER
    //______________________________________
    function writeCameraUniforms(){
        const aspect = canvas.width / canvas.height;

        //forward, right, up vectors
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
            frame, progressive ? 1 : 0, bgFlag, mattShader,
            ...sunDir, 0.0,
            ...sunRadiance, envScale
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
        writeCameraUniforms();

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
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

        requestAnimationFrame(render);
    }
    
    //_______________________________________
    // UI
    //_______________________________________

    //zoom wheel
    window.addEventListener('wheel', (e) => {
        cam_const *= 1.0 + 2.5e-4 * e.deltaY;
        requestAnimationFrame(render);
    }, {passive:false});

    //zoom slider
    document.getElementById('zoom').addEventListener('input', (e) => {
        cam_const = parseFloat(e.target.value);
        requestAnimationFrame(render);
    });

    //gamma slider
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

    document.getElementById("bgToggle").addEventListener("change", ev => {
        bgFlag = ev.target.checked ? 1.0 : 0.0;
        frame = 0;
        writeCameraUniforms();
    });

    document.getElementById("mattShader").addEventListener("change", (e) => {
        mattShader = parseFloat(e.target.value);
        frame = 0;
        requestAnimationFrame(render);
    });

    //_______________________________________________
    // FIRST FRAME
    //_____________________________________________
    updateJitters();
    requestAnimationFrame(render);
}
