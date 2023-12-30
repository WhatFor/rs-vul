pub mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/deferred.vert"
    }
}

pub mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/deferred.frag"
    }
}

pub mod lighting_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting.vert"
    }
}

pub mod lighting_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting.frag"
    }
}
