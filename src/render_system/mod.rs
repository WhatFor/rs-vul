pub mod lighting;
pub mod model;
pub mod obj_loader;
pub mod vp;

use nalgebra_glm::{half_pi, look_at, perspective, vec3};

use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;

use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSetsCollection,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::{ClearValue, Format},
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        self, AcquireError, PresentMode, Surface, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};

use crate::{
    render_system::lighting::{ambient::AmbientLight, directional::DirectionalLight},
    render_system::obj_loader::Vert,
    render_system::{model::Model, obj_loader::DummyVertex, vp::VP},
    shaders::{
        ambient_frag, ambient_vert, deferred_frag,
        deferred_vert::{self, VPData},
        directional_frag, directional_vert, light_obj_frag, light_obj_vert,
    },
};

use self::obj_loader::ColoredVertex;

// todo: Refactor
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
                depth_stencil: { depth },
                input: [ color, normals ],
            }
        ]
    )
    .unwrap()
}

// todo: Refactor
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

// todo: Refactor
pub fn build_deferred_pipeline(
    device: Arc<Device>,
    deferred_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let deferred_vert_s = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag_s = deferred_frag::load(device.clone()).unwrap();
    GraphicsPipeline::start()
        .vertex_input_state(Vert::per_vertex())
        .vertex_shader(deferred_vert_s.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(deferred_frag_s.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_render_pass)
        .build(device.clone())
        .unwrap()
}

// todo: Refactor
pub fn build_directional_lighting_pipeline(
    device: Arc<Device>,
    lighting_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let directional_vert_s = directional_vert::load(device.clone()).unwrap();
    let directional_frag_s = directional_frag::load(device.clone()).unwrap();
    GraphicsPipeline::start()
        .vertex_input_state(DummyVertex::per_vertex())
        .vertex_shader(directional_vert_s.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(directional_frag_s.entry_point("main").unwrap(), ())
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
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_render_pass)
        .build(device.clone())
        .unwrap()
}

// todo: Refactor
pub fn build_ambient_lighting_pipeline(
    device: Arc<Device>,
    lighting_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let ambient_vert_s = ambient_vert::load(device.clone()).unwrap();
    let ambient_frag_s = ambient_frag::load(device.clone()).unwrap();
    GraphicsPipeline::start()
        .vertex_input_state(DummyVertex::per_vertex())
        .vertex_shader(ambient_vert_s.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(ambient_frag_s.entry_point("main").unwrap(), ())
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
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_render_pass)
        .build(device.clone())
        .unwrap()
}

// todo: Refactor
pub fn build_lighting_obj_pipeline(
    device: Arc<Device>,
    lighting_render_pass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let light_obj_vert_s = light_obj_vert::load(device.clone()).unwrap();
    let light_obj_frag_s = light_obj_frag::load(device.clone()).unwrap();
    GraphicsPipeline::start()
        .vertex_input_state(ColoredVertex::per_vertex())
        .vertex_shader(light_obj_vert_s.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            viewport.clone()
        ]))
        .fragment_shader(light_obj_frag_s.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_render_pass.clone())
        .build(device.clone())
        .unwrap()
}

const TITLE: &str = "RS VUL";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const UNCAPPED_FPS: bool = true;

const NEAR_CLIP: f32 = 0.01;
const FAR_CLIP: f32 = 100.0;

pub struct RenderSystem {
    pub constants: EngineConstants,
    pub render_stage: RenderStage,

    instance: Arc<Instance>,
    window: Arc<Window>,
    surface: Arc<Surface>,
    viewport: Viewport,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>,
    framebuffers: Vec<Arc<Framebuffer>>,

    render_pass: Arc<RenderPass>,
    deferred_render_pass: Subpass,
    lighting_render_pass: Subpass,

    deferred_pipeline: Arc<GraphicsPipeline>,
    ambient_lighting_pipeline: Arc<GraphicsPipeline>,
    directional_lighting_pipeline: Arc<GraphicsPipeline>,
    light_obj_pipeline: Arc<GraphicsPipeline>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,

    vp_buffer_allocator: SubbufferAllocator,
    vp_buffer: Subbuffer<VPData>,

    model_buffer_allocator: SubbufferAllocator,

    ambient_buffer_allocator: SubbufferAllocator,
    ambient_light_buffer: Subbuffer<ambient_frag::AmbientLight>,

    directional_buffer_allocator: SubbufferAllocator,

    dummy_vertex_buffer: Subbuffer<[DummyVertex]>,

    colour_buffer: Arc<ImageView<AttachmentImage>>,
    normal_buffer: Arc<ImageView<AttachmentImage>>,
    depth_buffer: Arc<ImageView<AttachmentImage>>,

    vp: VP,
    vp_descriptor_set: Arc<PersistentDescriptorSet>,

    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    current_image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
}

pub struct EngineConstants {
    /// For each attachment in our Render Pass that has a load operation of LoadOp::Clear,
    /// the clear values that should be used for the attachments in the framebuffer.
    /// There must be exactly framebuffer.attachments().len() elements provided,
    /// and each one must match the attachment format.
    clear_values: Vec<Option<ClearValue>>,

    /// The extensions we want to make use of within Vulkan.
    default_vulkan_extensions: DeviceExtensions,

    /// The default, global illumination light.
    ambient_light: AmbientLight,
}

#[derive(Debug, Clone)]
pub enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    LightObject,
    NeedsRedraw,
}

impl RenderSystem {
    ///
    /// A standard Black colour for generic Clear Values.
    ///
    const GLOBAL_CLEAR_COLOUR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    ///
    /// Colour and intensity of the global illumination light.
    ///
    const AMBIENT_LIGHT_INTENSITY: f32 = 0.2;
    const AMBIENT_LIGHT_COLOUR: [f32; 3] = [1.0, 1.0, 1.0];

    ///
    /// The state our application starts in.
    ///
    const INITIAL_RENDER_STAGE: RenderStage = RenderStage::Stopped;

    ///
    /// Create a new MainApplication, including a Vulkan Instance and a winit EventLoop.
    /// This also initialises a lot of our constant values.
    ///
    pub fn new(event_loop: &EventLoop<()>) -> (Self, Option<Box<dyn GpuFuture>>) {
        let library = VulkanLibrary::new().expect("Failed to load vulkan library");
        let required_extensions = vulkano_win::required_extensions(&library);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let constants = EngineConstants {
            clear_values: vec![
                Some(Self::GLOBAL_CLEAR_COLOUR.into()), // colour, 0
                Some(Self::GLOBAL_CLEAR_COLOUR.into()), // normal, 1
                Some(Self::GLOBAL_CLEAR_COLOUR.into()), // uniform, 2
                Some(1.0.into()),
            ],
            default_vulkan_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::empty()
            },
            ambient_light: AmbientLight {
                color: Self::AMBIENT_LIGHT_COLOUR,
                intensity: Self::AMBIENT_LIGHT_INTENSITY,
            },
        };

        let surface = WindowBuilder::new()
            .with_title(TITLE)
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        let mut vp = VP::new();

        // Uniforms
        vp.view = look_at(
            &vec3(0.0, 0.0, 0.1),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        );

        // Calc Projection
        let image_extent: [u32; 2] = window.inner_size().into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        vp.projection = perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

        let (physical_device, queue_index) = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .filter(|device| {
                device
                    .supported_extensions()
                    .contains(&constants.default_vulkan_extensions)
            })
            .filter_map(|device| {
                device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(index, queue_props)| {
                        queue_props.queue_flags.contains(QueueFlags::GRAPHICS)
                            && device
                                .surface_support(index as u32, &surface)
                                .unwrap_or(false)
                    })
                    .map(|queue_index| (device, queue_index as u32))
            })
            .min_by_key(|(device, _)| match device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("No device found");

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_index,
                    ..Default::default()
                }],
                enabled_extensions: constants.default_vulkan_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let image_format = Some(
                physical_device
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            let window_dimensions = window.inner_size();

            let device_capabilities = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("Failed to get surface capabilities.");

            let alpha = device_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap();

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: device_capabilities.min_image_count + 1,
                    image_format,
                    image_extent: window_dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: alpha,
                    present_mode: if UNCAPPED_FPS {
                        PresentMode::Immediate
                    } else {
                        PresentMode::Fifo
                    },
                    ..Default::default()
                },
            )
            .unwrap()
        };

        // Allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        let vp_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let vp_buffer = {
            let uniform_data = deferred_vert::VPData {
                view: vp.view.into(),
                proj: vp.projection.into(),
            };

            let uniform_buffer: Subbuffer<deferred_vert::VPData> =
                vp_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let model_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_usage: MemoryUsage::Upload,
                ..Default::default()
            },
        );

        let ambient_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let directional_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let ambient_light_buffer = {
            let uniform_data = ambient_frag::AmbientLight {
                color: constants.ambient_light.color.into(),
                intensity: constants.ambient_light.intensity.into(),
            };

            let uniform_buffer: Subbuffer<ambient_frag::AmbientLight> =
                ambient_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let render_pass = get_render_pass(&device, &swapchain);
        let deferred_render_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_render_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let dummy_vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let image_dimensions = images[0].dimensions().width_height();

        // Renderpass buffers
        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(&memory_allocator, image_dimensions, Format::D16_UNORM)
                .unwrap(),
        )
        .unwrap();

        let colour_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &memory_allocator,
                image_dimensions,
                Format::A2B10G10R10_UNORM_PACK32,
            )
            .unwrap(),
        )
        .unwrap();

        let normal_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &memory_allocator,
                image_dimensions,
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers = gen_framebuffers(
            &images,
            &render_pass,
            &depth_buffer,
            &colour_buffer,
            &normal_buffer,
        );

        let deferred_pipeline = build_deferred_pipeline(
            device.clone(),
            deferred_render_pass.clone(),
            viewport.clone(),
        );

        let directional_lighting_pipeline = build_directional_lighting_pipeline(
            device.clone(),
            lighting_render_pass.clone(),
            viewport.clone(),
        );

        let ambient_lighting_pipeline = build_ambient_lighting_pipeline(
            device.clone(),
            lighting_render_pass.clone(),
            viewport.clone(),
        );

        let light_obj_pipeline = build_lighting_obj_pipeline(
            device.clone(),
            lighting_render_pass.clone(),
            viewport.clone(),
        );

        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();

        let vp_descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
        )
        .unwrap();

        let previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

        let rs = RenderSystem {
            instance,
            window,
            surface,
            viewport,
            queue,
            device,
            swapchain,
            images,
            framebuffers,

            render_pass,
            deferred_render_pass,
            lighting_render_pass,

            deferred_pipeline,
            directional_lighting_pipeline,
            ambient_lighting_pipeline,
            light_obj_pipeline,

            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,

            vp_buffer,
            vp_buffer_allocator,

            model_buffer_allocator,

            ambient_light_buffer,
            ambient_buffer_allocator,

            directional_buffer_allocator,

            dummy_vertex_buffer,

            depth_buffer,
            colour_buffer,
            normal_buffer,

            vp,
            vp_descriptor_set,

            constants,

            render_stage: Self::INITIAL_RENDER_STAGE,
            commands: None,
            current_image_index: 0,
            acquire_future: None,
        };

        (rs, previous_frame_end)
    }

    ///
    /// Create a new CommandBufferBuilder.
    ///
    pub fn create_command_builder(
        &self,
        allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Unable to create Command Buffer Builder!")
    }

    ///
    ///
    ///
    pub fn attach_framebuffer_to_command_builder(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &Arc<Framebuffer>,
    ) {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: self.constants.clear_values.clone(),
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap();
    }

    ///
    ///
    ///
    pub fn bind_pipeline_to_command_builder<S>(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        descriptor_sets: S,
    ) where
        S: DescriptorSetsCollection,
    {
        builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_sets,
            );
    }

    ///
    ///
    ///
    pub fn draw_vertex_buffer_within_command_builer<T>(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        vertex_buffer: &Subbuffer<[T]>,
    ) {
        builder
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .expect("Failed to draw vertex buffer!");
    }

    ///
    /// Given a DirectionalLight, generate a buffer containing the light information.
    ///
    fn generate_directional_light_buffer(
        &self,
        allocator: &SubbufferAllocator,
        light: &DirectionalLight,
    ) -> Subbuffer<directional_frag::DirectionalLight> {
        let uniform_data = directional_frag::DirectionalLight {
            position: light.position.into(),
            colour: light.color.into(),
        };

        let uniform_buffer: Subbuffer<directional_frag::DirectionalLight> =
            allocator.allocate_sized().unwrap();

        *uniform_buffer.write().unwrap() = uniform_data;

        uniform_buffer
    }

    pub fn set_ambient_light(&mut self, color: [f32; 3], intensity: f32) {
        let uniform_buffer: Subbuffer<ambient_frag::AmbientLight> =
            self.ambient_buffer_allocator.allocate_sized().unwrap();

        *uniform_buffer.write().unwrap() = ambient_frag::AmbientLight { color, intensity };

        self.ambient_light_buffer = uniform_buffer;
    }

    pub fn start_frame(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            }
            RenderStage::NeedsRedraw => {
                println!("Application is in invalid state. Stopping...");
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                println!("Application is in invalid state. Stopping...");
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {e}"),
            };

        if suboptimal {
            println!("Suboptimal Swapchain. Recreate next frame.");
            self.recreate_swapchain();
        }

        let mut command_buffer =
            self.create_command_builder(&self.command_buffer_allocator, &self.queue);

        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: self.constants.clear_values.clone(),
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap();

        self.commands = Some(command_buffer);
        self.current_image_index = image_index;
        self.acquire_future = Some(acquire_future);
    }

    pub fn finish_frame(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        match self.render_stage {
            RenderStage::Directional => {}
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                println!("Application is in invalid state. Stopping...");
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass().unwrap();
        let command_buffer = commands.build().unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        std::mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.current_image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }

    pub fn add_geometry(&mut self, model: &mut Model) {
        match self.render_stage {
            RenderStage::Deferred => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let vertex_buffer = Buffer::from_iter(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            model.data().iter().cloned(),
        )
        .unwrap();

        let model_uniform_buffer = {
            let (model_mat, normal_mat) = model.model_matrices();

            let uniform_data = deferred_vert::ModelData {
                model: model_mat.into(),
                normals: normal_mat.into(),
            };

            let uniform_buffer: Subbuffer<deferred_vert::ModelData> =
                self.model_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let model_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();

        let model_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            [WriteDescriptorSet::buffer(0, model_uniform_buffer.clone())],
        )
        .unwrap();

        let mut commands = self.commands.take().unwrap();

        self.bind_pipeline_to_command_builder(
            &mut commands,
            &self.deferred_pipeline,
            (self.vp_descriptor_set.clone(), model_descriptor_set.clone()),
        );

        self.draw_vertex_buffer_within_command_builer(&mut commands, &vertex_buffer);

        self.commands = Some(commands);
    }

    pub fn draw_ambient_light(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                // Finished the Deferred stage. Okay to move to Ambient.
                self.render_stage = RenderStage::Ambient;
            }
            RenderStage::Ambient => {
                // We've already called this. Return.
                return;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let ambient_lighting_layout = self
            .ambient_lighting_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        let ambient_lighting_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            ambient_lighting_layout.clone(),
            [
                // register all bindings for shader uniform buffers
                // index param == binding value in shader.
                WriteDescriptorSet::image_view(0, self.colour_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.ambient_light_buffer.clone()),
            ],
        )
        .unwrap();

        let mut commands = self.commands.take().unwrap();

        commands.next_subpass(SubpassContents::Inline).unwrap();

        self.bind_pipeline_to_command_builder(
            &mut commands,
            &self.ambient_lighting_pipeline,
            ambient_lighting_descriptor_set.clone(),
        );

        self.draw_vertex_buffer_within_command_builer(&mut commands, &self.dummy_vertex_buffer);

        self.commands = Some(commands);
    }

    pub fn draw_directional_light(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                // We've finished the Ambient pass, so OK to move to the Directional.
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {
                // We're already in Directinal mode. This is okay - we can draw multiples.
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let directional_light_subbuffer = self.generate_directional_light_buffer(
            &self.directional_buffer_allocator,
            &directional_light,
        );

        let directional_lighting_layout = self
            .directional_lighting_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        let directional_lighting_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            directional_lighting_layout.clone(),
            [
                // register all bindings for shader uniform buffers
                // index param == binding value in shader
                WriteDescriptorSet::image_view(0, self.colour_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, directional_light_subbuffer.clone()),
            ],
        )
        .unwrap();

        let mut commands = self.commands.take().unwrap();

        self.bind_pipeline_to_command_builder(
            &mut commands,
            &self.directional_lighting_pipeline,
            directional_lighting_descriptor_set.clone(),
        );

        // We need to draw the dummy vertex buffer so the lighting
        // pass only applies once to every pixel on the image.
        // Otherwise, overlapping and backfacing faces will be lit
        // and lead to an odd transparent look to complex objects.
        self.draw_vertex_buffer_within_command_builer(&mut commands, &self.dummy_vertex_buffer);

        self.commands = Some(commands);
    }

    pub fn draw_light_object(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Directional => {
                self.render_stage = RenderStage::LightObject;
            }
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        // TODO: This is essentially hard-coded; Would be wise to do this in `new()`
        let mut model = Model::new("resources/models/sphere.obj")
            .color(directional_light.color)
            .uniform_scale_factor(0.2)
            .build();

        // Move our sphere model to the position of our light
        model.translate(directional_light.get_position());

        let model_buffer = {
            let (model_mat, normal_mat) = model.model_matrices();

            let uniform_data = light_obj_vert::ModelData {
                model: model_mat.into(),
                normals: normal_mat.into(),
            };

            // TODO: This is strictly the wrong Subbuffer type - move to light_obj_vert::ModelData
            let uniform_buffer: Subbuffer<light_obj_vert::ModelData> =
                self.model_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let model_layout = self
            .light_obj_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();

        let model_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            [WriteDescriptorSet::buffer(0, model_buffer.clone())],
        )
        .unwrap();

        let vertex_buffer = Buffer::from_iter(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            model.color_data().iter().cloned(),
        )
        .unwrap();

        let mut commands = self.commands.take().unwrap();

        self.bind_pipeline_to_command_builder(
            &mut commands,
            &self.light_obj_pipeline,
            (self.vp_descriptor_set.clone(), model_set.clone()),
        );

        self.draw_vertex_buffer_within_command_builer(&mut commands, &vertex_buffer);

        self.commands = Some(commands);
    }

    pub fn recreate_swapchain(&mut self) {
        let new_dimensions = self.window.inner_size();

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {e}"),
        };

        self.swapchain = new_swapchain;
        self.images = new_images;
        self.render_pass = get_render_pass(&self.device, &self.swapchain);

        self.deferred_render_pass = Subpass::from(self.render_pass.clone(), 0).unwrap();
        self.lighting_render_pass = Subpass::from(self.render_pass.clone(), 1).unwrap();

        let new_depth_dimensions = self.images[0].dimensions().width_height();

        self.depth_buffer = ImageView::new_default(
            AttachmentImage::transient(
                &self.memory_allocator,
                new_depth_dimensions,
                Format::D16_UNORM,
            )
            .unwrap(),
        )
        .unwrap();

        self.colour_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &self.memory_allocator,
                new_depth_dimensions,
                Format::A2B10G10R10_UNORM_PACK32,
            )
            .unwrap(),
        )
        .unwrap();

        self.normal_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &self.memory_allocator,
                new_depth_dimensions,
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        self.framebuffers = gen_framebuffers(
            &self.images,
            &self.render_pass,
            &self.depth_buffer,
            &self.colour_buffer,
            &self.normal_buffer,
        );

        self.viewport.dimensions = new_dimensions.into();
        let image_extent: [u32; 2] = new_dimensions.into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;

        // Calc Projection
        self.vp.projection = perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

        let new_vp_buffer = {
            let uniform_data = deferred_vert::VPData {
                view: self.vp.view.into(),
                proj: self.vp.projection.into(),
            };

            let uniform_buffer: Subbuffer<deferred_vert::VPData> =
                self.vp_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let new_vp_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        self.vp_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            new_vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, new_vp_buffer.clone())],
        )
        .unwrap();

        self.deferred_pipeline = build_deferred_pipeline(
            self.device.clone(),
            self.deferred_render_pass.clone(),
            self.viewport.clone(),
        );

        self.directional_lighting_pipeline = build_directional_lighting_pipeline(
            self.device.clone(),
            self.lighting_render_pass.clone(),
            self.viewport.clone(),
        );

        self.ambient_lighting_pipeline = build_ambient_lighting_pipeline(
            self.device.clone(),
            self.lighting_render_pass.clone(),
            self.viewport.clone(),
        );

        self.light_obj_pipeline = build_lighting_obj_pipeline(
            self.device.clone(),
            self.lighting_render_pass.clone(),
            self.viewport.clone(),
        );
    }
}
