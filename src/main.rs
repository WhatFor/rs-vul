use nalgebra_glm::{half_pi, look_at, perspective, pi, rotate_normalized_axis, vec3};

use std::{sync::Arc, time::Instant};
use vulkano_win::VkSurfaceBuild;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
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
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::{ClearValue, Format},
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::Instance,
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
        StandardMemoryAllocator,
    },
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline},
    render_pass::{Framebuffer, Subpass},
    swapchain::{
        self, AcquireError, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, future::FenceSignalFuture, FlushError, GpuFuture},
};

use rs_vul::{
    lighting::{ambient::AmbientLight, directional::DirectionalLight},
    model::Model,
    obj_loader::DummyVertex,
    shaders::{
        ambient_frag, ambient_vert, deferred_frag, deferred_vert, directional_frag,
        directional_vert,
    },
    vp::VP,
    vulkan::{
        build_deferred_pipeline, build_lighting_pipeline, create_new_vulkano_instance,
        gen_framebuffers, get_render_pass,
    },
};

const TITLE: &str = "RS VUL";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const UNCAPPED_FPS: bool = true;

const NEAR_CLIP: f32 = 0.01;
const FAR_CLIP: f32 = 100.0;

struct MainApplication {
    global_state: GlobalApplicationState,
    constants: ApplicationConstants,
}

struct GlobalApplicationState {
    /// The instance of Vulkan.
    /// Used to spawn the VkSurface and select a Phsyical Device, not much else.
    instance: Arc<Instance>,
}

struct ApplicationConstants {
    /// For each attachment in our Render Pass that has a load operation of LoadOp::Clear,
    /// the clear values that should be used for the attachments in the framebuffer.
    /// There must be exactly framebuffer.attachments().len() elements provided,
    /// and each one must match the attachment format.
    clear_values: Vec<Option<ClearValue>>,

    /// The extensions we want to make use of within Vulkan.
    default_vulkan_extensions: DeviceExtensions,

    /// The default, global illumination light.
    ambient_light: AmbientLight,

    /// The start time of the application.
    start_time: Instant,
}

impl MainApplication {
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
    /// Create a new MainApplication, including a Vulkan Instance and a winit EventLoop.
    /// This also initialises a lot of our constant values.
    ///
    pub fn new() -> Self {
        let global_state = GlobalApplicationState {
            instance: create_new_vulkano_instance(),
        };

        let constants = ApplicationConstants {
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
            start_time: Instant::now(),
        };

        MainApplication {
            global_state,
            constants,
        }
    }

    ///
    /// Create a winit Window and Surface.
    ///
    fn spawn_surface_and_window(&self, event_loop: &EventLoop<()>) -> (Arc<Surface>, Arc<Window>) {
        let surface = WindowBuilder::new()
            .with_title(TITLE)
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .build_vk_surface(&event_loop, self.global_state.instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        (surface, window)
    }

    ///
    /// Select a Physical Device from our Instance, ensuring the device:
    ///  - supports all of our required Vulkan Extensions,
    ///  - contains at least one Graphics-enabled QueueFamily
    /// and prioritising selecting a device via the following hierarchy:
    ///  - Discrete GPU, Integrated GPU, Virtual GPU, CPU.
    ///
    fn select_physical_device(&self, surface: &Arc<Surface>) -> (Arc<PhysicalDevice>, u32) {
        let (physical_device, queue_index) = self
            .global_state
            .instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .filter(|device| {
                device
                    .supported_extensions()
                    .contains(&self.constants.default_vulkan_extensions)
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

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        (physical_device, queue_index)
    }

    pub fn create_logical_device(
        &self,
        physical_device: &Arc<PhysicalDevice>,
        queue_family_index: u32,
    ) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
        let (device, queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: self.constants.default_vulkan_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        return (device, queues);
    }

    ///
    /// Get:
    ///  - The Physical Device we want to use,
    ///  - A logical Device representing the Physical Device,
    ///  - A default Queue for our Device.
    ///
    fn get_devices(
        &self,
        surface: &Arc<Surface>,
    ) -> (Arc<PhysicalDevice>, Arc<Device>, Arc<Queue>) {
        let (physical_device, queue_family_index) = self.select_physical_device(&surface);

        let (logical_device, mut queues) =
            self.create_logical_device(&physical_device, queue_family_index);

        let default_device_queue = queues.next().unwrap();

        (physical_device, logical_device, default_device_queue)
    }

    ///
    /// Build common, long-lived Allocators.
    ///
    fn build_standard_allocators(
        &self,
        device: &Arc<Device>,
    ) -> (
        Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        StandardDescriptorSetAllocator,
        StandardCommandBufferAllocator,
    ) {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        (
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
        )
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
            // todo: try this out!
            //.set_viewport(0, [viewport.clone()])
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
    ///
    ///
    pub fn end_and_build_command_builder(
        &self,
        mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        builder
            .end_render_pass()
            .expect("Failed to end Render Pass!");

        Arc::new(builder.build().unwrap())
    }

    ///
    /// Build our Swapchain.
    ///
    fn build_swapchain(
        &self,
        physical_device: &Arc<PhysicalDevice>,
        window: &Arc<Window>,
        device: &Arc<Device>,
        surface: &Arc<Surface>,
    ) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
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

        let composite_alpha = device_capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();

        println!("Building Swapchain. Format: {:?}", image_format);

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: device_capabilities.min_image_count + 1,
                image_format,
                image_extent: window_dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                present_mode: if UNCAPPED_FPS {
                    PresentMode::Immediate
                } else {
                    PresentMode::Fifo
                },
                ..Default::default()
            },
        )
        .unwrap();

        println!(
            "Swapchain created. Image count: {:?}",
            swapchain.image_count()
        );

        (swapchain, images)
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

    ///
    /// The main entry point for our Application.
    ///
    pub fn run(self) {
        let event_loop = EventLoop::new();

        let (surface, window) = self.spawn_surface_and_window(&event_loop);
        let (physical_device, device, default_device_queue) = self.get_devices(&surface);

        let (mut swapchain, images) =
            self.build_swapchain(&physical_device, &window, &device, &surface);

        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            self.build_standard_allocators(&device);

        // Create shaders
        let deferred_vert_s = deferred_vert::load(device.clone()).expect("Failed to compile VS");
        let deferred_frag_s = deferred_frag::load(device.clone()).expect("Failed to compile FS");

        let directional_vert_s =
            directional_vert::load(device.clone()).expect("Failed to compile VS");
        let directional_frag_s =
            directional_frag::load(device.clone()).expect("Failed to compile FS");

        let ambient_vert_s = ambient_vert::load(device.clone()).expect("Failed to compile VS");
        let ambient_frag_s = ambient_frag::load(device.clone()).expect("Failed to compile FS");
        println!("Built shaders.");

        // Uniforms
        let mut vp = VP::new();
        vp.view = look_at(
            &vec3(0.0, 0.0, 0.1),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        );

        // Calc Projection
        let image_extent: [u32; 2] = window.inner_size().into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        vp.projection = perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

        // todo: do I really need all these different allocators?
        let vp_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let model_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
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

        println!("Loading model...");
        let mut model = Model::new("resources/models/suzanne.obj")
            .invert_winding_order(true)
            .build();

        model.translate(vec3(0.0, 0.0, -2.0));

        println!("Model loaded. Transforming to vertex buffer...");
        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
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

        println!("Getting render pass...");
        let render_pass = get_render_pass(&device, &swapchain);

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let image_dimensions = images[0].dimensions().width_height();

        // Renderpass buffers
        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(&memory_allocator, image_dimensions, Format::D16_UNORM)
                .unwrap(),
        )
        .unwrap();

        let mut colour_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &memory_allocator,
                image_dimensions,
                Format::A2B10G10R10_UNORM_PACK32,
            )
            .unwrap(),
        )
        .unwrap();

        let mut normal_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                &memory_allocator,
                image_dimensions,
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        let mut framebuffers = gen_framebuffers(
            &images,
            &render_pass,
            &depth_buffer,
            &colour_buffer,
            &normal_buffer,
        );

        println!("Getting viewport...");
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        println!("Getting Deferred pipeline...");
        let mut deferred_pipeline = build_deferred_pipeline(
            device.clone(),
            deferred_vert_s.clone(),
            deferred_frag_s.clone(),
            deferred_pass,
            viewport.clone(),
        );

        println!("Getting Lighting pipelines...");
        let mut directional_lighting_pipeline = build_lighting_pipeline(
            device.clone(),
            directional_vert_s.clone(),
            directional_frag_s.clone(),
            lighting_pass.clone(),
            viewport.clone(),
        );

        let mut ambient_lighting_pipeline = build_lighting_pipeline(
            device.clone(),
            ambient_vert_s.clone(),
            ambient_frag_s.clone(),
            lighting_pass.clone(),
            viewport.clone(),
        );

        let vp_subbuffer = {
            let uniform_data = deferred_vert::VPData {
                view: vp.view.into(),
                proj: vp.projection.into(),
            };

            let uniform_buffer: Subbuffer<deferred_vert::VPData> =
                vp_buffer_allocator.allocate_sized().unwrap();

            *uniform_buffer.write().unwrap() = uniform_data;

            uniform_buffer
        };

        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();

        let mut vp_descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_subbuffer.clone())],
        )
        .unwrap();

        let mut window_resized = false;
        let mut recreate_swapchain = false;

        let frames_in_flight = images.len();

        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    window_resized = true;
                }
                Event::MainEventsCleared => {} // todo: should maybe move some stuff into here?
                // RedrawEventsCleared seems more for cleanup
                Event::RedrawEventsCleared => {
                    if recreate_swapchain {
                        recreate_swapchain = false;
                        let new_dimensions = window.inner_size();

                        let (new_swapchain, new_images) =
                            match swapchain.recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("Failed to recreate swapchain: {e}"),
                            };

                        swapchain = new_swapchain;

                        let new_render_pass = get_render_pass(&device, &swapchain);

                        let new_depth_dimensions = new_images[0].dimensions().width_height();
                        let new_depth_buffer = ImageView::new_default(
                            AttachmentImage::transient(
                                &memory_allocator,
                                new_depth_dimensions,
                                Format::D16_UNORM,
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        colour_buffer = ImageView::new_default(
                            AttachmentImage::transient_input_attachment(
                                &memory_allocator,
                                new_depth_dimensions,
                                Format::A2B10G10R10_UNORM_PACK32,
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        normal_buffer = ImageView::new_default(
                            AttachmentImage::transient_input_attachment(
                                &memory_allocator,
                                new_depth_dimensions,
                                Format::R16G16B16A16_SFLOAT,
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        framebuffers = gen_framebuffers(
                            &new_images,
                            &new_render_pass,
                            &new_depth_buffer,
                            &colour_buffer,
                            &normal_buffer,
                        );

                        if window_resized {
                            window_resized = false;
                            println!("Window resized. Recrating pipeline...");

                            viewport.dimensions = new_dimensions.into();
                            let image_extent: [u32; 2] = new_dimensions.into();
                            let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;

                            // Calc Projection
                            vp.projection =
                                perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

                            let new_vp_subbuffer = {
                                let uniform_data = deferred_vert::VPData {
                                    view: vp.view.into(),
                                    proj: vp.projection.into(),
                                };

                                let uniform_buffer: Subbuffer<deferred_vert::VPData> =
                                    vp_buffer_allocator.allocate_sized().unwrap();

                                *uniform_buffer.write().unwrap() = uniform_data;

                                uniform_buffer
                            };

                            let new_vp_layout =
                                deferred_pipeline.layout().set_layouts().get(0).unwrap();

                            vp_descriptor_set = PersistentDescriptorSet::new(
                                &descriptor_set_allocator,
                                new_vp_layout.clone(),
                                [WriteDescriptorSet::buffer(0, new_vp_subbuffer.clone())],
                            )
                            .unwrap();

                            // todo: these following pipelines, do they need to be re-created??

                            // todo: redefining deferred pass here, bit shit
                            let deferred_pass = Subpass::from(new_render_pass.clone(), 0).unwrap();
                            deferred_pipeline = build_deferred_pipeline(
                                device.clone(),
                                deferred_vert_s.clone(),
                                deferred_frag_s.clone(),
                                deferred_pass.clone(),
                                viewport.clone(),
                            );

                            // todo: redefining lighting pass here, bit shit
                            let lighting_pass = Subpass::from(new_render_pass.clone(), 1).unwrap();
                            directional_lighting_pipeline = build_lighting_pipeline(
                                device.clone(),
                                directional_vert_s.clone(),
                                directional_frag_s.clone(),
                                lighting_pass.clone(),
                                viewport.clone(),
                            );

                            ambient_lighting_pipeline = build_lighting_pipeline(
                                device.clone(),
                                ambient_vert_s.clone(),
                                ambient_frag_s.clone(),
                                lighting_pass.clone(),
                                viewport.clone(),
                            );
                        }
                    }

                    let (image_index, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {e}"),
                        };

                    let model_subbuffer = {
                        let elapsed = self.constants.start_time.elapsed().as_secs() as f64
                            + self.constants.start_time.elapsed().subsec_nanos() as f64
                                / 1_000_000_000.0;

                        let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;

                        model.zero_rotation();
                        model.rotate(pi(), vec3(0.0, 1.0, 0.0));
                        model.rotate(elapsed_as_radians as f32 * 50.0, vec3(0.0, 0.0, 1.0));
                        model.rotate(elapsed_as_radians as f32 * 30.0, vec3(0.0, 1.0, 0.0));
                        model.rotate(elapsed_as_radians as f32 * 20.0, vec3(1.0, 0.0, 0.0));

                        let (model_mat, normal_mat) = model.model_matrices();

                        let uniform_data = deferred_vert::ModelData {
                            model: model_mat.into(),
                            normals: normal_mat.into(),
                        };

                        let uniform_buffer: Subbuffer<deferred_vert::ModelData> =
                            model_buffer_allocator.allocate_sized().unwrap();

                        *uniform_buffer.write().unwrap() = uniform_data;

                        uniform_buffer
                    };

                    let model_layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();

                    let model_descriptor_set = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        model_layout.clone(),
                        [WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
                    )
                    .unwrap();

                    let ambient_light_subbuffer = {
                        let uniform_data = ambient_frag::AmbientLight {
                            color: self.constants.ambient_light.color.into(),
                            intensity: self.constants.ambient_light.intensity.into(),
                        };

                        let uniform_buffer: Subbuffer<ambient_frag::AmbientLight> =
                            ambient_buffer_allocator.allocate_sized().unwrap();

                        *uniform_buffer.write().unwrap() = uniform_data;

                        uniform_buffer
                    };

                    let red_light = DirectionalLight {
                        position: [-4.0, 0.0, -4.0, 1.0],
                        color: [1.0, 0.0, 0.0],
                    };

                    let directional_light_subbuffer_r = self.generate_directional_light_buffer(
                        &directional_buffer_allocator,
                        &red_light,
                    );

                    let green_light = DirectionalLight {
                        position: [0.0, -4.0, 1.0, 1.0],
                        color: [0.0, 1.0, 0.0],
                    };

                    let directional_light_subbuffer_g = self.generate_directional_light_buffer(
                        &directional_buffer_allocator,
                        &green_light,
                    );

                    let blue_light = DirectionalLight {
                        position: [4.0, -2.0, 1.0, 1.0],
                        color: [0.0, 0.0, 1.0],
                    };

                    let directional_light_subbuffer_b = self.generate_directional_light_buffer(
                        &directional_buffer_allocator,
                        &blue_light,
                    );

                    let ambient_lighting_layout = ambient_lighting_pipeline
                        .layout()
                        .set_layouts()
                        .get(0)
                        .unwrap();

                    let ambient_lighting_descriptor_set = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        ambient_lighting_layout.clone(),
                        [
                            // register all bindings for shader uniform buffers
                            // index param == binding value in shader -
                            WriteDescriptorSet::image_view(0, colour_buffer.clone()),
                            WriteDescriptorSet::buffer(1, ambient_light_subbuffer.clone()),
                        ],
                    )
                    .unwrap();

                    // Only need this once for all directional lights
                    let directional_lighting_layout = directional_lighting_pipeline
                        .layout()
                        .set_layouts()
                        .get(0)
                        .unwrap();

                    // RED LIGHT
                    let directional_lighting_descriptor_set_r = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        directional_lighting_layout.clone(),
                        [
                            // register all bindings for shader uniform buffers
                            // index param == binding value in shader
                            WriteDescriptorSet::image_view(0, colour_buffer.clone()),
                            WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                            WriteDescriptorSet::buffer(2, directional_light_subbuffer_r.clone()),
                        ],
                    )
                    .unwrap();

                    // GREEN LIGHT
                    let directional_lighting_descriptor_set_g = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        directional_lighting_layout.clone(),
                        [
                            // register all bindings for shader uniform buffers
                            // index param == binding value in shader
                            WriteDescriptorSet::image_view(0, colour_buffer.clone()),
                            WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                            WriteDescriptorSet::buffer(2, directional_light_subbuffer_g.clone()),
                        ],
                    )
                    .unwrap();

                    // BLUE LIGHT
                    let directional_lighting_descriptor_set_b = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        directional_lighting_layout.clone(),
                        [
                            // register all bindings for shader uniform buffers
                            // index param == binding value in shader
                            WriteDescriptorSet::image_view(0, colour_buffer.clone()),
                            WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                            WriteDescriptorSet::buffer(2, directional_light_subbuffer_b.clone()),
                        ],
                    )
                    .unwrap();

                    // Build Command Buffer!
                    let command_buffer = {
                        let mut builder = self.create_command_builder(
                            &command_buffer_allocator,
                            &default_device_queue,
                        );

                        self.attach_framebuffer_to_command_builder(
                            &mut builder,
                            &framebuffers[image_index as usize],
                        );

                        // Deferred Pipeline
                        self.bind_pipeline_to_command_builder(
                            &mut builder,
                            &deferred_pipeline,
                            (vp_descriptor_set.clone(), model_descriptor_set.clone()),
                        );

                        self.draw_vertex_buffer_within_command_builer(&mut builder, &vertex_buffer);

                        // Finish Deferred Subpass, Start Lighting Subpass
                        // Note: all lighting steps occur on the same subpass, so we don't
                        //       call this again after each lighting pipeline.
                        builder.next_subpass(SubpassContents::Inline).unwrap();

                        // Directional Lighting - RED
                        self.bind_pipeline_to_command_builder(
                            &mut builder,
                            &directional_lighting_pipeline,
                            directional_lighting_descriptor_set_r.clone(),
                        );

                        // We need to draw the dummy vertex buffer so the lighting
                        // pass only applies once to every pixel on the image.
                        // Otherwise, overlapping and backfacing faces will be lit
                        // and lead to an odd transparent look to complex objects.
                        self.draw_vertex_buffer_within_command_builer(
                            &mut builder,
                            &dummy_vertex_buffer,
                        );

                        // Directional Lighting - GREEN
                        self.bind_pipeline_to_command_builder(
                            &mut builder,
                            &directional_lighting_pipeline,
                            directional_lighting_descriptor_set_g.clone(),
                        );

                        self.draw_vertex_buffer_within_command_builer(
                            &mut builder,
                            &dummy_vertex_buffer,
                        );

                        // Directional Lighting - BLUE
                        self.bind_pipeline_to_command_builder(
                            &mut builder,
                            &directional_lighting_pipeline,
                            directional_lighting_descriptor_set_b.clone(),
                        );

                        self.draw_vertex_buffer_within_command_builer(
                            &mut builder,
                            &dummy_vertex_buffer,
                        );

                        // Ambient Lighting
                        self.bind_pipeline_to_command_builder(
                            &mut builder,
                            &ambient_lighting_pipeline,
                            ambient_lighting_descriptor_set.clone(),
                        );

                        self.draw_vertex_buffer_within_command_builer(
                            &mut builder,
                            &dummy_vertex_buffer,
                        );

                        self.end_and_build_command_builder(builder)
                    };

                    if suboptimal {
                        println!("Suboptimal Swapchain. Recreate next frame.");
                        recreate_swapchain = true;
                    }

                    // wait for the fence related to this image to finish (normally this would be the oldest fence)
                    if let Some(image_fence) = &fences[image_index as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i as usize].clone() {
                        // Create a NowFuture
                        None => {
                            let mut now = sync::now(device.clone());
                            now.cleanup_finished();
                            now.boxed()
                        }
                        // Use the existing FenceSignalFuture
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(default_device_queue.clone(), command_buffer.clone())
                        .unwrap()
                        .then_swapchain_present(
                            default_device_queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                swapchain.clone(),
                                image_index,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_index as usize] = match future {
                        Ok(value) => Some(Arc::new(value)),
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            None
                        }
                    };

                    previous_fence_i = image_index;
                }
                _ => {}
            }
        });
    }
}

fn main() {
    let application = MainApplication::new();
    application.run();
}
