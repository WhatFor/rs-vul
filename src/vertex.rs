use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(Clone, Copy, Debug, Default, Vertex, Zeroable, Pod)]
#[repr(C)]
pub struct Vert {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub colour: [f32; 3],
}
