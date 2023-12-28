pub mod gl_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/gl.vert"
    }
}

pub mod gl_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/gl.frag"
    }
}
