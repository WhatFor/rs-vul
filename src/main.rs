use nalgebra_glm::{
    half_pi, identity, look_at, perspective, pi, rotate_normalized_axis, translate, vec3,
};

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
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
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
    mvp::MVP,
    obj,
    shaders::{deferred_frag, deferred_vert, lighting_frag, lighting_vert},
    vertex::Vert,
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

const AMBIENT_LIGHT_INTENSITY: f32 = 0.2;
const AMBIENT_LIGHT_COLOUR: [f32; 3] = [1.0, 1.0, 1.0];

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
}

impl MainApplication {
    ///
    /// A standard Black colour for generic Clear Values.
    ///
    const GLOBAL_CLEAR_COLOUR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

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
    /// Build the CommandBuffer.
    ///
    pub fn get_command_buffer(
        &self,
        allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        deferred_pipeline: &Arc<GraphicsPipeline>,
        lighting_pipeline: &Arc<GraphicsPipeline>,
        framebuffer: &Arc<Framebuffer>,
        vertex_buffer: &Subbuffer<[Vert]>,
        deferred_descriptor_set: &Arc<PersistentDescriptorSet>,
        lighting_descriptor_set: &Arc<PersistentDescriptorSet>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary(
            allocator,
            queue.queue_family_index(),
            // todo: we're creating and using this in the render loop, so kinda fine for OTS, but keep in mind.
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: self.constants.clear_values.clone(),
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap()
            // todo: try this out!
            //.set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(deferred_pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                deferred_pipeline.layout().clone(),
                0,
                deferred_descriptor_set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(lighting_pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                lighting_pipeline.layout().clone(),
                0,
                lighting_descriptor_set.clone(),
            )
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();

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
    /// The main entry point for our Application.
    ///
    pub fn run(self) {
        // Initial setup: Create Devices, surface, Swapchain, etc.
        let event_loop = EventLoop::new();

        let (surface, window) = self.spawn_surface_and_window(&event_loop);

        let (physical_device, queue_family_index) = self.select_physical_device(&surface);

        let (device, mut queues) = self.create_logical_device(&physical_device, queue_family_index);

        let default_device_queue = queues.next().unwrap();

        let (mut swapchain, images) =
            self.build_swapchain(&physical_device, &window, &device, &surface);

        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            self.build_standard_allocators(&device);

        // Create shaders
        let deferred_vert_s = deferred_vert::load(device.clone()).expect("Failed to compile VS");
        let deferred_frag_s = deferred_frag::load(device.clone()).expect("Failed to compile FS");
        let lighting_vert_s = lighting_vert::load(device.clone()).expect("Failed to compile VS");
        let lighting_frag_s = lighting_frag::load(device.clone()).expect("Failed to compile FS");
        println!("Built shaders.");

        // Uniforms
        let mut mvp = MVP::new();
        mvp.model = translate(&identity(), &vec3(0.0, 0.0, -2.0));
        mvp.view = look_at(
            &vec3(0.0, 0.0, 0.1),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        );

        let ambient_light = AmbientLight {
            color: AMBIENT_LIGHT_COLOUR,
            intensity: AMBIENT_LIGHT_INTENSITY,
        };

        // testing a directional light:
        let dir_light = DirectionalLight {
            color: [1.0, 1.0, 1.0],
            position: [-4.0, -4.0, 0.0, 1.0],
        };

        // For rotating obj
        let rotation_start = Instant::now();

        let uniform_buffer_allocator: SubbufferAllocator = SubbufferAllocator::new(
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
        let obj_data = obj::load_model("resources/models/duck.obj");

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
            obj_data.verticies,
        )
        .unwrap();

        println!("Getting render pass...");
        let mut render_pass = get_render_pass(&device, &swapchain);

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

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

        println!("Getting Lighting pipeline...");
        let mut lighting_pipeline = build_lighting_pipeline(
            device.clone(),
            lighting_vert_s.clone(),
            lighting_frag_s.clone(),
            lighting_pass,
            viewport.clone(),
        );

        let mut image_extent: [u32; 2] = window.inner_size().into();
        let mut aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;

        // Calc Projection
        mvp.projection = perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

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
                        println!("Recrating swapchain...");

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

                        render_pass = get_render_pass(&device, &swapchain);

                        // todo: all these buffers are duplicated above
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

                        //todo: should this be default? tut said 'new'
                        let new_colour_buffer = ImageView::new_default(
                            AttachmentImage::transient_input_attachment(
                                &memory_allocator,
                                new_depth_dimensions,
                                Format::A2B10G10R10_UNORM_PACK32,
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        //todo: same here
                        let new_normal_buffer = ImageView::new_default(
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
                            &render_pass,
                            &new_depth_buffer,
                            &new_colour_buffer,
                            &new_normal_buffer,
                        );

                        if window_resized {
                            window_resized = false;
                            println!("Window resized. Recrating pipeline...");

                            viewport.dimensions = new_dimensions.into();
                            // todo; warnings?
                            image_extent = window.inner_size().into();
                            aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;

                            // Calc Projection
                            mvp.projection =
                                perspective(aspect_ratio, half_pi(), NEAR_CLIP, FAR_CLIP);

                            // todo: redefining deferred pass here, bit shit
                            let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
                            deferred_pipeline = build_deferred_pipeline(
                                device.clone(),
                                deferred_vert_s.clone(),
                                deferred_frag_s.clone(),
                                deferred_pass.clone(),
                                viewport.clone(),
                            );

                            // todo: redefining lighting pass here, bit shit
                            let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();
                            lighting_pipeline = build_lighting_pipeline(
                                device.clone(),
                                lighting_vert_s.clone(),
                                lighting_frag_s.clone(),
                                lighting_pass,
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

                    // Write to our uniform buffers - this largely controls the rotation of the duck.
                    let uniform_buffer_subbuffer = {
                        // Calc Model Matrix
                        let elapsed = rotation_start.elapsed().as_secs() as f64
                            + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                        let elapsed_as_radians = elapsed * pi::<f64>() / 180.0 * 30.0;
                        let model = rotate_normalized_axis(
                            &mvp.model,
                            elapsed_as_radians as f32,
                            &vec3(1.0, 0.0, 0.0),
                        );

                        let double_rotate = rotate_normalized_axis(
                            &model,
                            elapsed_as_radians as f32,
                            &vec3(0.0, 1.0, 0.0),
                        );

                        let uniform_data = deferred_vert::UniformBufferObject {
                            model: double_rotate.into(),
                            view: (mvp.view).into(),
                            proj: mvp.projection.into(),
                        };

                        // Write Uniform Buffers
                        let uniform_buffer: Subbuffer<deferred_vert::UniformBufferObject> =
                            uniform_buffer_allocator.allocate_sized().unwrap();

                        *uniform_buffer.write().unwrap() = uniform_data;

                        uniform_buffer
                    };

                    let ambient_light_subbuffer = {
                        let uniform_data = lighting_frag::AmbientLight {
                            colour: ambient_light.color.into(),
                            intensity: ambient_light.intensity.into(),
                        };

                        let uniform_buffer: Subbuffer<lighting_frag::AmbientLight> =
                            ambient_buffer_allocator.allocate_sized().unwrap();

                        *uniform_buffer.write().unwrap() = uniform_data;

                        uniform_buffer
                    };

                    let directional_light_subbuffer = {
                        let uniform_data = lighting_frag::DirectionalLight {
                            position: dir_light.position.into(),
                            colour: dir_light.color.into(),
                        };

                        let uniform_buffer: Subbuffer<lighting_frag::DirectionalLight> =
                            directional_buffer_allocator.allocate_sized().unwrap();

                        *uniform_buffer.write().unwrap() = uniform_data;

                        uniform_buffer
                    };

                    let deferred_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();

                    let deferred_descriptor_set = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        deferred_layout.clone(),
                        [WriteDescriptorSet::buffer(
                            0,
                            uniform_buffer_subbuffer.clone(),
                        )],
                    )
                    .unwrap();

                    let lighting_layout = lighting_pipeline.layout().set_layouts().get(0).unwrap();

                    let lighting_descriptor_set = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        lighting_layout.clone(),
                        [
                            // register all bindings for shader uniform buffers
                            // index param == binding value in shader
                            WriteDescriptorSet::image_view(0, colour_buffer.clone()),
                            WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                            WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer),
                            WriteDescriptorSet::buffer(3, ambient_light_subbuffer),
                            WriteDescriptorSet::buffer(4, directional_light_subbuffer),
                        ],
                    )
                    .unwrap();

                    let command_buffer = self.get_command_buffer(
                        &command_buffer_allocator,
                        &default_device_queue,
                        &deferred_pipeline,
                        &lighting_pipeline,
                        &framebuffers[image_index as usize],
                        &vertex_buffer,
                        &deferred_descriptor_set,
                        &lighting_descriptor_set,
                    );

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
