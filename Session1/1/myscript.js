"use strict";

/*
Definitions : 
- Canvas : a rectangular area in the web page where we can draw
- WebGPU : the web graphics API
- GPU : Graphics Processing Unit, the graphics card
- Shader : a small program that runs directly on the GPU
- Pipeline : a "plan" that tells the GPU how to draw things
- Render pass : a single drawing operation on the canvas
- Command buffer : a list of instructions for the GPU
*/


//we initialize graphics, load shaders, setup pipeline
window.onload = function() { main(); } //wait for page to load
async function main()
{
    //initialize webgpu
    const gpu = navigator.gpu; //access webgpu
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
    const wgslfile = document.getElementById('wgsl').src; //we get the URL of the shader file w1.wgsl
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({
        code: wgslcode
    });

    //Pipeline setup
    const pipeline = device.createRenderPipeline({
        layout: 'auto', //let WebGPU figure out ressource layout
        vertex: { module: wgsl, entryPoint: 'main_vs', }, //vertex stage
        fragment: { 
        module: wgsl,
        entryPoint: 'main_fs', 
        targets: [{ format: canvasFormat }], },
        primitive: { topology: 'triangle-strip', }, //we draw a strip of triangles
    });

    // Create a render pass in a command buffer and submit it
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(), 
            loadOp: "clear",
            storeOp: "store",
        }]
    });

    // Insert render pass commands here
    pass.setPipeline(pipeline); //we choose the pipeline
    pass.draw(4); //draw 4 vertices, enough for 2 triangles
    pass.end(); //end the render pass
    device.queue.submit([encoder.finish()]); //submit the command buffer
}