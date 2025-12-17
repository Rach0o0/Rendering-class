"use strict";

/*
Create a struct (HitInfo) for recording hit information and implement ray-plane intersection, raysphere intersection, and ray-triangle intersection. Call these intersection functions in the fragment
shader to render the default scene (specified below). Assign the designated object colour to the pixel
when a ray intersects it. Use the light blue rectangle colour from before as background colour if
no intersection was found. Adjust the maximum trace distance of your ray (tmax) if you have an
intersection to ensure that you end up with the closest intersection.
*/
window.onload = function() { main(); }
async function main()
{
    //we initialize graphics, load shaders, setup pipeline

    //initialize webgpu
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    //setup canvas
    const canvas = document.getElementById('my-canvas');
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({
        device: device,
        format: canvasFormat,
    });

    
    //load shader
    const wgslfile = document.getElementById('wgsl').src;
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({
        code: wgslcode
    });

    //Pipeline setup
    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: wgsl, entryPoint: 'main_vs', },
        fragment: {
        module: wgsl,
        entryPoint: 'main_fs',
        targets: [{ format: canvasFormat }], },
        primitive: { topology: 'triangle-strip', },
    });

    //Create uniform buffer and bind group
    let bytelength = 6*sizeof['vec4']; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
    }],
    });

    // Set uniform values
    const eye = vec3(2.0, 1.5, 2.0);
    let cam_const = 1.0;
    const p = vec3(0.0, 0.5, 0.0);
    const u = vec3(0.0, 1.0, 0.0);
    //let gamma = 2.2;

    let sphereShader = 3;
    let mattShader = 1;

    //funciton that computes camera basis and fills the uniform buffer
    function writeCameraUniforms(){
        const aspect = canvas.width / canvas.height;

        //forward, right, up vectors
        let v = normalize(subtract(p, eye));
        let b1 = normalize(cross(v, u));
        let b2 = cross(b1, v);

        let values = [
            aspect, cam_const, 0.0, 0.0,
            ...eye, 0.0,
            ...b1, 0.0,
            ...b2, 0.0,
            ...v, 0.0,
            sphereShader, mattShader, 0.0, 0.0,
        ];

        //console.log("Uniforms values:", values);

        //pack into 5 vec4 to match WGSL layout
        new Float32Array(uniforms, 0, 24).set(values);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    }

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
    /*document.getElementById("gamma-slider").addEventListener("input", (event) => {
        gamma = parseFloat(event.target.value);
        requestAnimationFrame(animate);
    });*/

    document.getElementById('sphereShader').addEventListener('change', (e) => {
    sphereShader = parseInt(e.target.value);
    requestAnimationFrame(render);
    });

    document.getElementById('mattShader').addEventListener('change', (e) => {
        mattShader = parseInt(e.target.value);
        requestAnimationFrame(render);
        render();
    });

    requestAnimationFrame(render);
}