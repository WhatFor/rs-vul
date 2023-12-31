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

pub mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting/directional.vert"
    }
}

pub mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting/directional.frag"
    }
}

pub mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting/ambient.vert"
    }
}

pub mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting/ambient.frag"
    }
}
