"use strict";

//when the page load, we call main
window.onload = function() { main(); }


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
    const obj = await readOBJFile("objects/CornellBoxWithBlocks.obj", 1, true);
    console.log("attribs length =", obj.attribs.length);       
    console.log("attribs =", obj.attribs);
    
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
    buffers.lightIndices = device.createBuffer({
        size: obj.light_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(buffers.lightIndices, 0, obj.light_indices);

    console.log("Light indices array:", obj.light_indices);
    for (let i = 0; i < obj.light_indices.length; i++) {
        console.log(`Light triangle ${i}:`, obj.light_indices[i]);
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
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }, 
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
        targets: [{ format: canvasFormat }], },
        primitive: { topology: 'triangle-strip', },
    });

    //______________________________________
    // UNIFORM BUFFER
    //______________________________________
    let bytelength = 6*16; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    //________________________________________
    // CREATE JITTER BUFFER
    //________________________________________

    //preallocate storage buffer to store an array of jitter vectors
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
                    const idx = (i * subdivs +j) * 2;
                    jitter[idx] = (Math.random() + j) * step - pixelsize * 0.5;
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
            { binding: 0,resource: { buffer: uniformBuffer }},
            { binding: 1, resource: {buffer: jitterBuffer}},
            { binding: 2, resource: { buffer: buffers.attribs } },
            { binding: 3, resource: { buffer: buffers.indices } },
            { binding: 4, resource: { buffer: buffers.materials } },
            { binding: 5, resource: { buffer: buffers.bspTree } },
            { binding: 6, resource: { buffer: buffers.treeIds } }, 
            { binding: 7, resource: { buffer: buffers.bspPlanes } },
            { binding: 8, resource: { buffer: buffers.aabb } }, 
            { binding: 9, resource: { buffer: buffers.lightIndices } },
        ],
    });
    
    //______________________________________
    // PARAMETERS
    //______________________________________

    //camera
    const eye = vec3(277.0, 275.0, -570.0);
    const p = vec3(277.0, 275.0, 0.0);
    const u = vec3(0.0, 1.0, 0.0);
    let cam_const = 1.0;
    let gamma = 2.2;



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
            ...b1, 0.0,
            ...b2, 0.0,
            ...v, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];

        //(debug) console.log("Uniforms values:", values);

        //pack into 6 vec4 to match WGSL layout
        const uniformData = new Float32Array(values);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
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
    
    //_______________________________________
    // UI
    //_______________________________________

    //zoom wheel
    window.addEventListener('wheel', (e) => {
        cam_const *= 1.0 + 2.5e-4*e.deltaY;
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
        requestAnimationFrame(render);
    });

    //UI
    /*
    document.getElementById('sphereShader').addEventListener('change', (e) => {
    sphereShader = parseInt(e.target.value);
    requestAnimationFrame(render);
    });
    */


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