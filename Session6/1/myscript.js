"use strict";

// when the page loads, call main
window.onload = function () { main(); }

async function main() {

    //______________________________________
    // INITIALIZATION
    //______________________________________
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageBuffersPerShaderStage: 10,
    },
    });

    //_____________________________________
    // LOAD 3D OBJECT
    //_____________________________________
    const obj = await readOBJFile("objects/CornellBoxWithBlocks.obj", 1, true);

    const buffers = {};
    build_bsp_tree(obj, device, buffers); // builds bspTree, treeIds, bspPlanes, aabb
    console.log("BSP buffers:", buffers);
    console.log("AABB:", buffers.aabb);

    // vertex positions
    const positionBuffer = device.createBuffer({
    size: obj.vertices.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(positionBuffer, 0, obj.vertices);

    // face indices
    const indexBuffer = device.createBuffer({
    size: obj.indices.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(indexBuffer, 0, obj.indices);

    // vertex normals
    const normalBuffer = device.createBuffer({
    size: obj.normals.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(normalBuffer, 0, obj.normals);

    //______________________________________
    // MATERIAL BUFFERS
    //______________________________________
    const matIndexBuffer = device.createBuffer({
    size: obj.mat_indices.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(matIndexBuffer, 0, obj.mat_indices);

    const materialCount = obj.materials.length;
    const materialStrideFloats = 8; // color(3)+pad + emission(3)+pad
    const materialData = new Float32Array(materialCount * materialStrideFloats);

    for (let i = 0; i < materialCount; i++) {
    const m = obj.materials[i];
    const kd = [m.color.r, m.color.g, m.color.b];
    const ke = [m.emission.r, m.emission.g, m.emission.b];
    const base = i * materialStrideFloats;
    materialData.set([...kd, 0.0, ...ke, 0.0], base);
    }

    const materialBuffer = device.createBuffer({
    size: materialData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(materialBuffer, 0, materialData);

    //______________________________________
    // LIGHT INDICES BUFFER
    //______________________________________
    const lightIndexBuffer = device.createBuffer({
    size: obj.light_indices.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(lightIndexBuffer, 0, obj.light_indices);

    console.log("Number of emissive triangles:", obj.light_indices.length);

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
    // LOAD SHADER
    //______________________________________
    const wgslfile = document.getElementById('wgsl').src;
    const wgslcode = await fetch(wgslfile, { cache: "reload" }).then(r => r.text());
    const wgsl = device.createShaderModule({ code: wgslcode });

    //______________________________________
    // CREATE RENDER PIPELINE
    //______________________________________
    const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
    });

    const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: wgsl, entryPoint: 'main_vs' },
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
    const uniformBuffer = device.createBuffer({
    size: 6 * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    //________________________________________
    // CREATE JITTER BUFFER
    //________________________________________
    let jitter = new Float32Array(200);
    const jitterBuffer = device.createBuffer({
    size: jitter.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });

    function compute_jitters(arr, pixelsize, subdivs) {
    const step = pixelsize / subdivs;
    if (subdivs < 2) {
        arr[0] = 0.0;
        arr[1] = 0.0;
    } else {
        for (let i = 0; i < subdivs; ++i) {
        for (let j = 0; j < subdivs; ++j) {
            const idx = (i * subdivs + j) * 2;
            arr[idx] = (Math.random() + j) * step - pixelsize * 0.5;
            arr[idx + 1] = (Math.random() + i) * step - pixelsize * 0.5;
        }
        }
    }
    }

    let subdivs = 1;
    function updateJitters() {
    const pixelsize = 1.0 / canvas.height;
    compute_jitters(jitter, pixelsize, subdivs);
    device.queue.writeBuffer(jitterBuffer, 0, jitter);
    }

    //______________________________________
    // BIND GROUP
    //______________________________________
    const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: jitterBuffer } },
        { binding: 2, resource: { buffer: positionBuffer } },
        { binding: 3, resource: { buffer: indexBuffer } },
        { binding: 4, resource: { buffer: normalBuffer } },
        { binding: 5, resource: { buffer: matIndexBuffer } },
        { binding: 6, resource: { buffer: materialBuffer } },
        { binding: 7, resource: { buffer: lightIndexBuffer } },
        { binding: 10, resource: { buffer: buffers.aabb } },
    ],
    });

    //______________________________________
    // CAMERA PARAMETERS
    //______________________________________
    const eye = vec3(277.0, 275.0, -570.0);
    const p = vec3(277.0, 275.0, 0.0);
    const u = vec3(0.0, 1.0, 0.0);
    let cam_const = 1.0;
    let gamma = 2.2;
    let sphereShader = 3;
    let mattShader = 1;

    function writeCameraUniforms() {
    const aspect = canvas.width / canvas.height;
    let v = normalize(subtract(p, eye));
    let b1 = normalize(cross(v, u));
    let b2 = cross(b1, v);
    let values = [
        aspect, cam_const, gamma, subdivs,
        ...eye, 0.0,
        ...b1, 0.0,
        ...b2, 0.0,
        ...v, 0.0,
        sphereShader, mattShader, 0.0, 0.0,
    ];
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array(values));
    }

    //_______________________________________
    // RENDER LOOP
    //_______________________________________
    const timer = new TimingHelper(device);
    let lastUpdate = performance.now();
    let accumulatedTime = 0; 
    let frameCount = 0;
    
    async function render(){
        const t0 = performance.now();

        writeCameraUniforms();

        const encoder = device.createCommandEncoder();
        const pass = timer.beginRenderPass(encoder,{
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
            }]
        });
        // Insert render pass commands here
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();

        device.queue.submit([encoder.finish()]);


        let durationMsGPU = 0;
        if (timer && timer.getResult) {
            try {
                const durationNs = await timer.getResult();
                durationMsGPU = durationNs / 1e6; // conversion ns → ms
            } catch {
                durationMsGPU = 0; // fallback si la feature n’est pas dispo
            }
        }

        const durationMsCPU = performance.now() - t0;

        // Si GPU = 0 (pas dispo), on prend CPU
        const frameTime = durationMsGPU > 0 ? durationMsGPU : durationMsCPU;

        accumulatedTime += frameTime;
        frameCount++;
        const now = performance.now();
        if (now - lastUpdate >= 1000) {
            const avgTime = accumulatedTime / frameCount;
            const fps = 1000 / avgTime;
            document.getElementById("frametime").textContent =
                `Frame time: ${avgTime.toFixed(2)} ms (${fps.toFixed(1)} FPS)`;

            accumulatedTime = 0;
            frameCount = 0;
            lastUpdate = now;
        }
        requestAnimationFrame(render);
    }

    //______________________________________
    // UI
    //______________________________________
    window.addEventListener('wheel', (e) => {
    cam_const *= 1.0 + 2.5e-4 * e.deltaY;
    requestAnimationFrame(render);
    }, { passive: false });

    document.getElementById('zoom').addEventListener('input', (e) => {
    cam_const = parseFloat(e.target.value);
    requestAnimationFrame(render);
    });

    document.getElementById("gamma-slider").addEventListener("input", (e) => {
    gamma = parseFloat(e.target.value);
    requestAnimationFrame(render);
    });

    document.getElementById('mattShader').addEventListener('change', (e) => {
    mattShader = parseInt(e.target.value);
    requestAnimationFrame(render);
    });

    document.getElementById("subdivs").addEventListener("input", (e) => {
    subdivs = parseInt(e.target.value);
    updateJitters();
    requestAnimationFrame(render);
    });

    //_______________________________________________
    // FIRST FRAME
    //_____________________________________________
    updateJitters();
    requestAnimationFrame(render);
}
