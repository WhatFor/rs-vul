use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::Swapchain,
    VulkanLibrary,
};

use crate::obj_loader::{DummyVertex, Vert};

pub fn create_new_vulkano_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("Failed to load vulkan library");
    let required_extensions = vulkano_win::required_extensions(&library);

    return Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Unable to fetch Vulkan Instance.");
}

pub fn get_render_pass(device: &Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            color: {
                load: Clear,
                store: DontCare,
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
            },
            normals: {
                load: Clear,
                store: DontCare,
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1
            }
        },
        passes: [
            {
                color: [ color, normals ],
                depth_stencil: { depth },
                input: [],
            },
            {
                color: [ final_color ],
                depth_stencil: {},
                input: [ color, normals ],
            }
        ]
    )
    .unwrap()
}

pub fn gen_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
    depth_buffer: &Arc<ImageView<AttachmentImage>>,
    colour_buffer: &Arc<ImageView<AttachmentImage>>,
    normal_buffer: &Arc<ImageView<AttachmentImage>>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        colour_buffer.clone(),
                        normal_buffer.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn build_deferred_pipeline(
    device: Arc<Device>,
    deferred_vert_shader: Arc<ShaderModule>,
    deferred_frag_shader: Arc<ShaderModule>,
    deferred_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(Vert::per_vertex())
        .vertex_shader(deferred_vert_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(deferred_frag_shader.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_render_pass)
        .build(device.clone())
        .unwrap()
}

pub fn build_lighting_pipeline(
    device: Arc<Device>,
    vert_s: Arc<ShaderModule>,
    frag_s: Arc<ShaderModule>,
    lighting_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(DummyVertex::per_vertex())
        .vertex_shader(vert_s.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(frag_s.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(lighting_render_pass.num_color_attachments()).blend(
                AttachmentBlend {
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                },
            ),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test()) // todo: is this needed?
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_render_pass)
        .build(device.clone())
        .unwrap()
}
