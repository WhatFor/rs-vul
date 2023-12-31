pub mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/deferred.vert"
    }
}

pub mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/deferred.frag"
    }
}

pub mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/lighting/directional.vert"
    }
}

pub mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/lighting/directional.frag"
    }
}

pub mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/lighting/ambient.vert"
    }
}

pub mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/lighting/ambient.frag"
    }
}
