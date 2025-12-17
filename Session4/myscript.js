"use strict";

//when the page load, we call main
window.onload = function() { main(); }

//load a texture
async function load_texture(device, filename)
{
    //download image
    const response = await fetch(filename);
    const blob = await response.blob();
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
    
    //format
    const texture = device.createTexture({
        size: [img.width, img.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });
    //CPU -> GPU
    device.queue.copyExternalImageToTexture(
        { source: img, flipY: true },
        { texture: texture },
        { width: img.width, height: img.height },
    );
    return texture;
}

async function main()
{
    //______________________________________
    // INITIALIZATION
    //______________________________________
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();

    //______________________________________
    // LOAD TEXTURE
    //______________________________________
    const texture = await load_texture(device, "grass.jpg");

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
    // PIPELINE SETUP
    //______________________________________
    const pipeline = device.createRenderPipeline({
        layout: 'auto',
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
    let bytelength = 7*sizeof['vec4']; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

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
    function makeBindGroup() {
        return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0,resource: { buffer: uniformBuffer }},
            { binding: 1, resource: texture.createView() },
            {binding: 2, resource: {buffer: jitterBuffer}}    
        ],
    });
    }
    let bindGroup= makeBindGroup();

    //______________________________________
    // PARAMETERS
    //______________________________________

    //camera
    const eye = vec3(2.0, 1.5, 2.0);
    let cam_const = 1.0;
    const p = vec3(0.0, 0.5, 0.0);
    const u = vec3(0.0, 1.0, 0.0);
    let gamma = 2.2;

    let sphereShader = 3;
    let mattShader = 1;

    //texture flags
    let useTexture = 1.0;
    let texScale = 1.0;
    let filterFlag = 0.0; // 0 = nearest, 1 = linear
    let repeatFlag = 1.0; // 1 = repeat, 0 = clamp

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
            useTexture, filterFlag, repeatFlag,texScale,
        ];

        //(debug) console.log("Uniforms values:", values);

        //pack into 6 vec4 to match WGSL layout
        new Float32Array(uniforms, 0, 28).set(values);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
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
        cam_const *= 1.0 + 2.5e-4*event.deltaY;
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

    document.getElementById('sphereShader').addEventListener('change', (e) => {
    sphereShader = parseInt(e.target.value);
    requestAnimationFrame(render);
    });

    document.getElementById('mattShader').addEventListener('change', (e) => {
        mattShader = parseInt(e.target.value);
        requestAnimationFrame(render);
        render();
    });

    document.getElementById('addrMode').addEventListener('change', (e) => {
        repeatFlag = (e.target.value === "repeat") ? 1.0 : 0.0;
        requestAnimationFrame(render);
    });

    document.getElementById('filterMode').addEventListener('change', (e) => {
        filterFlag = (e.target.value === "linear") ? 1.0 : 0.0;
        requestAnimationFrame(render);
    });

    document.getElementById('toggleTexture').addEventListener('change', (e) => {
        useTexture = e.target.checked ? 1.0 : 0.0;
        requestAnimationFrame(render);
    });

    document.getElementById("subdivs").addEventListener("input", (e) => {
    subdivs = parseInt(e.target.value);
    updateJitters();
    requestAnimationFrame(render);
    });

    document.getElementById('texScale').addEventListener('input', (e) => {
    texScale = parseFloat(e.target.value);
    requestAnimationFrame(render);
    });

    requestAnimationFrame(render);
}