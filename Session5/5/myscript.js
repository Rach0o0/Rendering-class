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
    const device = await adapter.requestDevice();

    //_____________________________________
    // LOAD 3D OBJECT
    //_____________________________________
    const obj = await readOBJFile("objects/CornellBoxWithBlocks.obj", 1, true); //scale = 1, reverse = true
    
    //vertex positions
    const positionBuffer = device.createBuffer({
        size: obj.vertices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(positionBuffer, 0, obj.vertices);

    //face indices
    const indexBuffer = device.createBuffer({
        size: obj.indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(indexBuffer, 0, obj.indices);
    
    //vertex normals
    const normalBuffer = device.createBuffer({
        size: obj.normals.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(normalBuffer, 0, obj.normals);
    
    console.log(obj.materials);

    //______________________________________
    // MATERIAL BUFFERS
    //______________________________________
    const matIndexBuffer = device.createBuffer({
        size: obj.mat_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(matIndexBuffer, 0, obj.mat_indices);

    const materialCount = obj.materials.length;
    const materialStrideFloats = 8; // color(3)+pad + emission(3)+pad
    const materialData = new Float32Array(materialCount * materialStrideFloats);

    for (let i = 0; i < materialCount; i++) {
        const m = obj.materials[i];
        const kd = [m.color.r, m.color.g, m.color.b];       // adapte selon tes champs
        const ke = [m.emission.r, m.emission.g, m.emission.b];    // adapte selon tes champs

        const base = i * materialStrideFloats;
        materialData[base + 0] = kd[0];
        materialData[base + 1] = kd[1];
        materialData[base + 2] = kd[2];
        materialData[base + 3] = 0.0; // pad
        materialData[base + 4] = ke[0];
        materialData[base + 5] = ke[1];
        materialData[base + 6] = ke[2];
        materialData[base + 7] = 0.0; // pad
    }

    const materialBuffer = device.createBuffer({
        size: materialData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(materialBuffer, 0, materialData);

    console.log(obj.materials);
    for (let m of obj.materials) {
        console.log(
            m.name.trim(),
            "color:", m.color,
            "emission:", m.emission
        );
    }

    //______________________________________
    // LIGHT INDICES BUFFER
    //______________________________________
    const lightIndexBuffer = device.createBuffer({
        size: obj.light_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
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
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({
        code: wgslcode
    });

    //______________________________________
    // CREATE RENDER PIPELINE
    //______________________________________
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            // 0: uniforms
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            // 1: jitter buffer (read-only)
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 2: vertex positions
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 3: triangle indices
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 4: vertex normals
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 5: material indices (u32 per triangle)
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 6: material properties (array<Material>)
            { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
            // 7: emissive triangle indices (array<u32>)
            { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
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
            { binding: 2, resource: { buffer: positionBuffer } },
            { binding: 3, resource: { buffer: indexBuffer } },
            { binding: 4, resource: { buffer: normalBuffer } },
            { binding: 5, resource: { buffer: matIndexBuffer } },
            { binding: 6, resource: { buffer: materialBuffer } }, 
            { binding: 7, resource: { buffer: lightIndexBuffer } },   
        ],
    });
    
    //______________________________________
    // PARAMETERS
    //______________________________________

    //camera
    const eye = vec3(277.0, 275.0, -570.0);
    let cam_const = 1.0;
    const p = vec3(277.0, 275.0, 0.0);
    const u = vec3(0.0, 1.0, 0.0);
    let gamma = 2.2;

    let sphereShader = 3;
    let mattShader = 1;

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
            sphereShader, mattShader, 0.0, 0.0,
        ];

        //(debug) console.log("Uniforms values:", values);

        //pack into 6 vec4 to match WGSL layout
        const uniformData = new Float32Array(values);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    }

    //_______________________________________
    // RENDER LOOP
    //_______________________________________
    function render(){
        writeCameraUniforms();

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
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
   
    document.getElementById('mattShader').addEventListener('change', (e) => {
        mattShader = parseInt(e.target.value);
        requestAnimationFrame(render);
        render();
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